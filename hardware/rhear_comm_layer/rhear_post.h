/* rhear_post.h -- post-processing that runs AFTER the mask: the
 * no-voice-no-sound gate and the FxNLMS adaptive line enhancer.
 * Split out so both can be validated on a host compiler. No Arduino here.
 */
#ifndef RHEAR_POST_H
#define RHEAR_POST_H
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ------------------------------------------------- gate: no voice, no sound */
/* This is why noise was still audible between words: it did not exist on the
 * device. gGateDb was -200 and both l1_frame() calls passed removeVoices =
 * false, so the only gate in the project lived in the host script.
 *
 * Threshold is measured against the SPEECH level -- the 90th-percentile frame
 * -- not against a percentile of everything. A percentile-of-everything
 * threshold follows the noise floor down after enhancement and stops closing,
 * which measured 0.0 dB improvement when it was tried that way. Against the
 * speech level it reaches 20% of frames below -70 dB on real material. */
#define GATE_FRAME    256
#define GATE_BELOW    10.0f     /* dB below the speech level.
   Swept on the real recording against the on-device DSP output: 20 dB closes
   only 7% of frames, 14 dB closes 8%, 10 dB closes 10%, 8 dB closes 20%. The
   DSP path leaves a higher residual floor than the neural model, so there is
   less dynamic range to gate on and the 20 dB that suited the neural output
   barely closes here. 10 dB is the compromise before quiet speech is at risk. */
#define GATE_PRE      3
#define GATE_POST     10
#define GATE_FLOOR    (-75.0f)

static int gateBuffer(int16_t *y, int n) {
  int m = n / GATE_FRAME; if (m < 8) return 0;
  static float e[1024]; if (m > 1024) m = 1024;
  for (int i = 0; i < m; i++) {
    double s = 0;
    for (int j = 0; j < GATE_FRAME; j++) { double v = y[i*GATE_FRAME+j]/32768.0; s += v*v; }
    e[i] = 10.0f * log10f((float)(s / GATE_FRAME) + 1e-20f);
  }
  /* 90th percentile without sorting the whole array */
  float lo = 1e30f, hi = -1e30f;
  for (int i = 0; i < m; i++) { if (e[i] < lo) lo = e[i]; if (e[i] > hi) hi = e[i]; }
  float spk = hi;
  for (int it = 0; it < 24; it++) {          /* bisect to the 90th pct */
    float mid = 0.5f*(lo+hi); int c = 0;
    for (int i = 0; i < m; i++) if (e[i] <= mid) c++;
    if (c*100 >= m*90) { spk = mid; hi = mid; } else lo = mid;
  }
  float thr = spk - GATE_BELOW;
  static uint8_t op[1024];
  for (int i = 0; i < m; i++) op[i] = e[i] > thr;
  static uint8_t dil[1024];
  int closed = 0;
  for (int i = 0; i < m; i++) {
    int a = i - GATE_PRE, b = i + GATE_POST, v = 0;
    if (a < 0) a = 0; if (b >= m) b = m - 1;
    for (int j = a; j <= b; j++) if (op[j]) { v = 1; break; }
    dil[i] = v; if (!v) closed++;
  }
  float fl = powf(10.0f, GATE_FLOOR/20.0f);
  for (int i = 0; i < m; i++) {                     /* short ramp, no clicks */
    int a = i-3, b = i+3, c = 0, t = 0;
    if (a < 0) a = 0; if (b >= m) b = m-1;
    for (int j = a; j <= b; j++) { c += dil[j]; t++; }
    float g = fl + (1.0f - fl) * ((float)c / t);
    for (int j = 0; j < GATE_FRAME; j++) y[i*GATE_FRAME+j] = (int16_t)(y[i*GATE_FRAME+j]*g);
  }
  return 100 * closed / m;
}

/* Streaming gate driven by the VAD, not by energy.
 *
 * The energy version compared each frame against a tracked speech level and
 * kept holding open, because energy cannot distinguish loud noise from a
 * voice -- exactly the failure reported. gVadLLR is the Sohn likelihood ratio
 * computed inside l1_frame from the a priori and a posteriori SNRs, which is
 * the statistic that CAN make that distinction. */
/* Measured on the real recording, separating speech frames from silent ones:
 *      raw frame energy          d' 5.11  -> 100.0% correct
 *      enhanced frame energy     d' 2.21  ->  99.7%
 *      enhanced above own floor  d' 2.03  ->  86.6%
 *      Sohn LLR                  d' 0.97  ->  77.8%
 * The simplest feature wins outright and the statistical VAD is the worst of
 * the four. The earlier energy gate did not fail because energy is the wrong
 * signal -- it failed because it measured against a tracked SPEECH level.
 * Measuring against the tracked NOISE FLOOR instead is what makes it work. */
/* Margins are small because they are relative to a floor that actually
 * tracks. The measured optimum on the real recording was an absolute -19.4 dB
 * against speech near -15 and quiet near -25, i.e. about 6 dB above the
 * floor. An earlier version used 12 dB above a floor seeded at -60 that rose
 * at 0.0015/frame -- a ten-second time constant -- so across a ten-second
 * clip the floor never reached the real one, "above floor" read ~30 dB
 * permanently, and the gate never closed once. Seeding the floor on the
 * first frame is what makes this work at all. */
/* Runtime-adjustable, because a threshold fitted to one recording is fitted
 * to one room. Tune by ear with 'G' / 'g' while watching the OPEN/-- marker
 * the live mode prints. */
static float gVadOpen  = 6.0f;   /* dB above the noise floor to open  */
static float gVadClose = 3.0f;   /* dB above to stay open: hysteresis */
static float gGateG = 0.0f, gFloor = 0.0f;
static bool  gFloorInit = false;
static int   gHang = 0;
static bool  gVoiced = false;

/* pass the RAW mic frame, not the enhanced one -- raw separates better */
float gLastAbove = 0.0f;      /* dB above floor, for the live display */
static void gateStream(int16_t *y, int n, const int16_t *raw) {
  double s = 0;
  for (int i = 0; i < n; i++) { double v = raw[i] / 32768.0; s += v * v; }
  float e = 10.0f * log10f((float)(s / n) + 1e-20f);
  if (!gFloorInit) { gFloor = e; gFloorInit = true; }   /* seed, do not crawl */
  /* fast down, slow up, so speech cannot drag the floor upward -- but the
   * up-rate is 0.01 (~1.6 s), not 0.0015 (~10 s), so it can follow a room
   * whose noise rises during the clip. */
  if (e < gFloor) gFloor += 0.25f  * (e - gFloor);
  else            gFloor += 0.01f  * (e - gFloor);
  float above = e - gFloor; gLastAbove = above;
  if (gVoiced) { if (above < gVadClose) gVoiced = false; }
  else         { if (above > gVadOpen)  gVoiced = true;  }
  if (gVoiced) gHang = GATE_POST; else if (gHang > 0) gHang--;
  float want = (gHang > 0) ? 1.0f : powf(10.0f, GATE_FLOOR/20.0f);
  /* asymmetric ramp: open fast so no word onset is clipped, close slowly so
   * the tail of a word is not chopped */
  gGateG += ((want > gGateG) ? 0.50f : 0.08f) * (want - gGateG);
  for (int i = 0; i < n; i++) y[i] = (int16_t)(y[i] * gGateG);
}

/* ------------------------------- FxNLMS as an adaptive line enhancer ------ */
/* The FxNLMS machinery in rhear_dsp.h is the L0 canceller. ACOUSTIC
 * cancellation needs the codec -- 146 us causality budget, and a digital mic
 * path cannot meet it -- so that is not what this is.
 *
 * What the SAME adaptive filter does usefully with one microphone and no
 * codec is remove PERIODIC noise from the comms signal: engine hum, rotor
 * blade rate, generator whine. Reference is the input delayed past speech's
 * own correlation length, so the filter can only predict the part that is
 * periodic over that delay -- the tone, not the voice. Subtracting the
 * prediction leaves speech. Classical adaptive line enhancer, and it comes
 * free with the taps already written and validated (+12.6 dB on tonal noise).
 *
 * It complements the neural mask rather than duplicating it: the mask is
 * trained on the statistics of noise, this tracks a specific tone as it
 * drifts in frequency. */
#define ALE_TAPS   192
#define ALE_DELAY  384         /* 24 ms at 16 kHz -- see the sweep below */
#define ALE_LEN    (ALE_TAPS + ALE_DELAY)
/* The decorrelation delay is the whole design. Swept on REAL speech plus a
 * 200/400 Hz tone (a sine used as "speech" is a pathological test -- an ALE
 * cannot tell a stationary sine from a tone, and the first sweep showed it
 * removing both equally):
 *      3 ms : tone -9.3 dB, speech -5.2 dB
 *     12 ms : tone -7.8 dB, speech -3.6 dB
 *     24 ms : tone -7.9 dB, speech -2.6 dB   <-- chosen
 *     48 ms : tone -6.8 dB, speech -2.1 dB
 * Below ~12 ms the delay is inside voiced speech's own correlation length, so
 * the filter predicts the voice and subtracts it. 24 ms is past that and
 * still well inside a steady tone's. Speech still pays 2.6 dB, which is why
 * this is OFF by default and toggled with 'F'. */
static float aleW[ALE_TAPS], aleX[ALE_LEN];
static int   alePos = 0;
static float alePow = 1.0f;
static bool  gAleOn = false;

static void aleReset(void) {
  memset(aleW, 0, sizeof aleW); memset(aleX, 0, sizeof aleX);
  alePos = 0; alePow = 1.0f;
}
/* circular, not shifting: a memmove of 576 floats per sample at 16 kHz is
 * ~37 MB/s of pure copying for no reason. */
static inline float aleStep(float d, float mu) {
  alePos = (alePos == 0) ? ALE_LEN - 1 : alePos - 1;
  aleX[alePos] = d;
  int j = (alePos + ALE_DELAY) % ALE_LEN;        /* decorrelated reference */
  float y = 0, p = 0; int k = j;
  for (int i = 0; i < ALE_TAPS; i++) {
    float v = aleX[k]; y += aleW[i] * v; p += v * v;
    if (++k == ALE_LEN) k = 0;
  }
  float e = d - y;                               /* periodic part removed  */
  alePow = 0.99f * alePow + 0.01f * p;
  float g = mu * e / (alePow + 1e-4f);
  k = j;
  for (int i = 0; i < ALE_TAPS; i++) {
    aleW[i] += g * aleX[k];
    if (++k == ALE_LEN) k = 0;
  }
  return e;
}


#endif /* RHEAR_POST_H */
