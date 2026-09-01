#!/usr/bin/env python3
"""Full RHEAR timing budget on the ESP32-S3 — can everything coexist?

The L0 result (E08) establishes one piece. This is the whole headset: L0, L1,
L2 and audio I/O against the real frame deadlines and the measured chip
ceiling.

Chip ceiling is Espressif's own published ESP-DSP O2 benchmark, not an
estimate: dsps_dotprod_s16 does 256 MACs in 307 cycles = 0.834 MAC/cycle, so
200 MMAC/s per core at 240 MHz, two cores.

IMPORTANT SCOPE NOTE. docs/02-architecture.md describes L2 as a neural scene
engine. The code in rhear/core/ is not that: it is classical feature extraction
(state.frame_features) plus a predictability measure (predict.prediction_floor_db)
plus filter selection (l2runtime). This budget prices WHAT EXISTS. A trained
neural L2 would need its own measurement and is not covered here.
"""
import math, json, os

# ---- chip ----
F_CPU = 240e6
MAC_PER_CYCLE_S16 = 0.834          # Espressif ESP-DSP O2, dsps_dotprod_s16
CORE = F_CPU * MAC_PER_CYCLE_S16 / 1e6      # MMAC/s per core
N_CORES = 2

# ---- L0: adaptive canceller ----
L0_FS, L0_L, L0_M, L0_CH = 192e3, 128, 64, 2
l0 = (3 * L0_L + L0_M) * L0_CH * L0_FS / 1e6

# ---- L1: speech enhancer (measured, phase branch removed) ----
l1 = 58.79                          # from E03/bench/model_diff.json
L1_FPS = 250.0                      # 4 ms hop
L1_FRAME_MS = 1000.0 / L1_FPS

# ---- L2: as implemented in rhear/core/ ----
L2_HZ = 62.5
L2_WIN = 1024                       # analysis window, samples
L2_ORDER = 256                      # predict.prediction_floor_db(order=256)

def fft_macs(n):                    # 5 n log2 n, the usual radix-2 count
    return 5 * n * math.log2(n)

# frame_features: one rfft + one irfft + O(n) statistics
l2_feat = (2 * fft_macs(L2_WIN) + 6 * L2_WIN) * L2_HZ / 1e6
# prediction_floor_db: acf via FFT, then solve a 256x256 Toeplitz system.
# AS WRITTEN it builds the full matrix and calls np.linalg.solve -> O(n^3)/3.
l2_solve_generic = (L2_ORDER ** 3 / 3) * L2_HZ / 1e6
# Levinson-Durbin exploits the Toeplitz structure for the SAME answer -> O(n^2).
l2_solve_levinson = (2 * L2_ORDER ** 2) * L2_HZ / 1e6
l2_acf = fft_macs(2 * L2_WIN) * L2_HZ / 1e6
l2_now = l2_feat + l2_acf + l2_solve_generic
l2_fixed = l2_feat + l2_acf + l2_solve_levinson

# ---- audio I/O ----
# L1's STFT/iSTFT is already inside its 58.79 (19.5% of it).
# L0 runs in the time domain, so its I/O is DMA + interrupt servicing, not MACs.
# Budgeted as cycles, converted to a MAC-equivalent so one table can add up.
L0_ISR_CYCLES = 40                  # per stereo sample pair, DMA-serviced in blocks
io = L0_ISR_CYCLES * L0_FS * MAC_PER_CYCLE_S16 / 1e6

rows = [
    ("L0  FxNLMS, L=128, stereo @192 kHz", l0, "core 0"),
    ("L1  speech enhancer @250 Hz (4 ms hop)", l1, "core 1"),
    ("L2  features + ACF @62.5 Hz", l2_feat + l2_acf, "core 1"),
    ("L2  predictability solve — AS WRITTEN", l2_solve_generic, "core 1"),
    ("audio I/O — DMA/ISR, MAC-equivalent", io, "core 0"),
]

def show(l2_solve, tag):
    c0 = l0 + io
    c1 = l1 + l2_feat + l2_acf + l2_solve
    tot = c0 + c1
    print("\n  === %s ===" % tag)
    print("  %-44s%12s%10s" % ("component", "MMAC/s", "core"))
    print("  " + "-" * 66)
    for name, v, core in rows[:3]:
        print("  %-44s%12.1f%10s" % (name, v, core))
    print("  %-44s%12.1f%10s" % ("L2  predictability solve", l2_solve, "core 1"))
    print("  %-44s%12.1f%10s" % (rows[4][0], io, "core 0"))
    print("  " + "-" * 66)
    print("  %-44s%12.1f%10s   %s" % ("core 0 total", c0, "",
          "PASS" if c0 <= CORE else "FAIL"))
    print("  %-44s%12.1f%10s   %s" % ("core 1 total", c1, "",
          "PASS" if c1 <= CORE else "FAIL"))
    print("  %-44s%12.1f" % ("TOTAL", tot))
    print("  %-44s%12.1f  (%.0f MMAC/s x %d cores)"
          % ("chip ceiling", CORE * N_CORES, CORE, N_CORES))
    print("  %-44s%11.0f%%" % ("utilisation", 100 * tot / (CORE * N_CORES)))
    ok = c0 <= CORE and c1 <= CORE
    print("  %-44s%12s" % ("FRAME DEADLINE", "PASS" if ok else "FAIL"))
    return dict(core0=c0, core1=c1, total=tot, pass_=bool(ok))

print("  ESP32-S3, %d cores at %.0f MHz, %.3f MAC/cycle int16 -> %.0f MMAC/s per core"
      % (N_CORES, F_CPU / 1e6, MAC_PER_CYCLE_S16, CORE))
print("  deadlines: L0 every %.1f us (192 kHz) | L1 every %.1f ms | L2 every %.1f ms"
      % (1e6 / L0_FS, L1_FRAME_MS, 1000 / L2_HZ))

a = show(l2_solve_generic, "BEFORE THE FIX — generic O(n^3) solver, for the record")
b = show(l2_solve_levinson, "AS THE CODE IS WRITTEN TODAY (Levinson-Durbin, fixed)")

print("\n  The difference is one function, not one chip.")
print("  prediction_floor_db builds a %dx%d Toeplitz matrix and calls a generic"
      % (L2_ORDER, L2_ORDER))
print("  solver: O(n^3)/3 = %.2f M ops every 16 ms = %.0f MMAC/s on its own."
      % (L2_ORDER ** 3 / 3 / 1e6, l2_solve_generic))
print("  Levinson-Durbin returns the SAME answer in O(n^2) = %.0f k ops = %.1f MMAC/s."
      % (2 * L2_ORDER ** 2 / 1e3, l2_solve_levinson))
print("  Ratio %.0fx. Without it the headset does not fit; with it there is room."
      % (l2_solve_generic / l2_solve_levinson))

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
json.dump({"chip": {"cores": N_CORES, "mmac_s_per_core": CORE},
           "as_written": a, "with_levinson": b,
           "components": {"l0": l0, "l1": l1, "l2_features": l2_feat + l2_acf,
                          "l2_solve_generic": l2_solve_generic,
                          "l2_solve_levinson": l2_solve_levinson, "io": io}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "timing_budget.json"), "w"), indent=1)
