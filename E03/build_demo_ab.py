#!/usr/bin/env python3
"""Rebuild the demo A/B pack (audio + summary.json) from a checkpoint.

    python3 build_demo_ab.py --ckpt runs/g2_fullband/best.pt

This did not exist as a script before, which is why the demo sat on
runs/cond_B/best.pt -- two model generations behind -- long after better
checkpoints landed. The pack was originally produced by an inline run that
could not be repeated. Now it can.

The example clip IDs are REUSED from the existing summary.json by default, so
a rebuild changes the model and nothing else. If the clip selection changed
too, an A/B would be comparing two different things at once and no listener
could tell which change they were hearing.
"""
import os, sys, json, argparse, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import FS
from train_interim import enhance, Pairs, DEV, count_params, count_macs_per_frame, HOP
from rhear_data.manifest import read_manifest
from eval_stratified import load_model, score_one, agg, SNR_BUCKETS

AB = os.path.join(HERE, "runs", "demo_ab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--out", default=AB)
    ap.add_argument("--reselect", action="store_true",
                    help="pick fresh example clips instead of reusing the "
                         "existing IDs (changes what listeners hear)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    meta = [r for r in rows if r["split"] == "test"]
    assert len(ds) == len(meta), "manifest/dataset mismatch"

    model, (has_phase, has_fb) = load_model(a.ckpt)
    params = count_params(model)
    macs, _ = count_macs_per_frame(model)
    print("  ckpt %s" % a.ckpt)
    print("  %d params | phase %s | full-band %s | %.2f MMAC/s"
          % (params, "PRESENT" if has_phase else "REMOVED",
             "PRESENT" if has_fb else "absent", macs * (FS / HOP) / 1e6))

    ck_hop = int(model.hop_.item())
    if ck_hop != HOP:
        print("  NOTE: checkpoint hop %d (module default %d) -- enhance() uses "
              "the model's own hop, so this is correct, but the A/B pack will "
              "be built at %d." % (ck_hop, HOP, ck_hop))
    win = torch.hann_window(512, device=DEV)
    scored, sig = [], {}
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        c, n = y.numpy(), x.numpy()
        e = est[0].cpu().numpy()
        L = min(len(c), len(n), len(e))
        r = dict(meta[i]); r.update(score_one(c[:L], n[:L], e[:L]))
        scored.append(r)
        sig[r["id"]] = (c[:L], n[:L], e[:L])
        if (i + 1) % 60 == 0:
            print("    %d/%d" % (i + 1, len(ds)), flush=True)

    ov = agg(scored)

    # ---- example clips: reuse the existing IDs unless told otherwise -------
    prev_path = os.path.join(a.out, "summary.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else {}
    if prev.get("examples") and not a.reselect:
        want = [(e["base"], e["id"]) for e in prev["examples"]]
        print("  reusing %d example clips from the existing pack" % len(want))
    else:
        by = {}
        for r in scored:
            for l in r.get("noise_layers", []):
                by.setdefault(l["cls"], []).append(r)
        want = []
        for i, cls in enumerate(sorted(by)):
            best = max(by[cls], key=lambda r: r["stoi_e"] - r["stoi_n"])
            want.append(("ex%02d" % i, best["id"]))
        print("  selected %d fresh example clips" % len(want))

    byid = {r["id"]: r for r in scored}
    examples = []
    for base, cid in want:
        if cid not in byid:
            print("    WARNING: %s not in this eval set, skipped" % cid); continue
        r = byid[cid]; c, n, e = sig[cid]
        for tag, s in (("clean", c), ("noisy", n), ("enhanced", e)):
            sf.write(os.path.join(a.out, "%s_%s.wav" % (base, tag)),
                     np.clip(s, -1, 1), FS)
        cls = [l["cls"] for l in r.get("noise_layers", [])]
        examples.append(dict(
            base=base, id=cid, snr_db=r.get("snr_db"),
            primary=cls[0] if cls else "?", classes=cls,
            reverb=r.get("reverb", "?"),
            stoi_noisy=round(r["stoi_n"], 3), stoi_enhanced=round(r["stoi_e"], 3),
            pesq_noisy=round(r["pesq_n"], 2), pesq_enhanced=round(r["pesq_e"], 2)))

    def strat(keyfn, keys):
        o = {}
        for k in keys:
            sel = [r for r in scored if keyfn(r, k)]
            b = agg(sel)
            if b:
                o[k] = dict(n=b["n"], d_stoi=round(b["stoi_e"] - b["stoi_n"], 3),
                            d_pesq=round(b["pesq_e"] - b["pesq_n"], 2),
                            d_sisdr=round(b["sisdr_e"] - b["sisdr_n"], 1))
        return o

    classes = sorted({l["cls"] for r in scored for l in r.get("noise_layers", [])})
    out = dict(prev)          # keep WARNING / provenance / audio_from
    out.update(
        checkpoint=os.path.relpath(a.ckpt, HERE),
        n_test=len(scored),
        model=dict(params=params, mmacs=round(macs * (FS / HOP) / 1e6, 3),
                   latency_ms=8.0,
                   phase_branch="removed" if not has_phase else "present",
                   fullband_branch="present" if has_fb else "absent"),
        # KEY NAMES ARE A UI CONTRACT. ui/index.html reads stoi_noisy /
        # stoi_enhanced / pesq_* / sisdr_*. agg() returns stoi_n / stoi_e, and
        # writing those through renamed the fields out from under the page:
        # the A/B panel got undefined for every number and stopped rendering,
        # which looked exactly like the audio files had been deleted. Emit the
        # names the page reads, and add the SI-SIR/SI-SAR split alongside.
        overall=dict(
            stoi_noisy=round(ov["stoi_n"], 4), stoi_enhanced=round(ov["stoi_e"], 4),
            pesq_noisy=round(ov["pesq_n"], 4), pesq_enhanced=round(ov["pesq_e"], 4),
            sisdr_noisy=round(ov["sisdr_n"], 3), sisdr_enhanced=round(ov["sisdr_e"], 3),
            si_sir=round(ov["sir_e"], 2), si_sar=round(ov["sar_e"], 2)),
        deltas=dict(d_stoi=round(ov["stoi_e"] - ov["stoi_n"], 4),
                    d_pesq=round(ov["pesq_e"] - ov["pesq_n"], 4),
                    d_sisdr=round(ov["sisdr_e"] - ov["sisdr_n"], 3)),
        by_class=strat(lambda r, k: any(l["cls"] == k for l in r.get("noise_layers", [])), classes),
        by_snr=strat(lambda r, k: k[0] <= r.get("snr_db", -999) < k[1],
                     [(lo, hi) for lo, hi, _ in SNR_BUCKETS]),
        examples=examples)
    out["by_snr"] = {lab: out["by_snr"][(lo, hi)]
                     for lo, hi, lab in SNR_BUCKETS if (lo, hi) in out["by_snr"]}

    f = out.setdefault("findings", {})
    f["headline"] = (
        "We are not failing to remove noise. We remove %.1f dB of it. "
        "Artefacts cap the net result at %.1f dB."
        % (ov["sir_e"], ov["sar_e"]))
    f["sir_sar_split"] = dict(
        si_sir_noise_removed_db=round(ov["sir_e"], 2),
        si_sar_damage_caused_db=round(ov["sar_e"], 2),
        si_sdr_net_db=round(ov["sisdr_e"], 2),
        ceiling_si_sar_db=22.91,
        note="SI-SDR alone cannot separate 'removed noise' from 'damaged "
             "speech'. Splitting it is what showed the bottleneck is "
             "distortion, not residue.")
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    print("\n  STOI  %.3f -> %.3f   PESQ %.3f -> %.3f"
          % (ov["stoi_n"], ov["stoi_e"], ov["pesq_n"], ov["pesq_e"]))
    print("  SI-SDR %.2f dB | SI-SIR %.2f | SI-SAR %.2f" % (ov["sisdr_e"], ov["sir_e"], ov["sar_e"]))
    print("  wrote %d examples + summary.json -> %s" % (len(examples), a.out))


if __name__ == "__main__":
    main()
