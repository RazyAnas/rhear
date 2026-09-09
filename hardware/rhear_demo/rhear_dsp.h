/* rhear_dsp.h -- the signal processing, split out so it can be tested on a
 * host compiler instead of only on the target. rhear_demo.ino includes this;
 * test_dsp.c links it and asserts the algorithms actually work. Nothing in
 * here may touch Arduino, FreeRTOS or I2S.
 */
#ifndef RHEAR_DSP_H
#define RHEAR_DSP_H
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define FS48        48000
#define L1_HOP      256
#define L1_NFFT     512
#define L0_TAPS     128
#define SHAT_TAPS   32

#ifndef RHEAR_HOST_TEST
extern volatile float gAlpha;
extern volatile float gGateDb;
#else
extern float gAlpha, gGateDb;
#endif

/* ------------------------------------------------------------ L0: FxNLMS */
/* y[n]   = w . x[n]                 anti-noise
 * e[n]   = d[n] - y_through_S[n]    residual at the (virtual) error mic
 * xf[n]  = Shat * x[n]              filtered reference
 * w     += mu * e * xf / (||xf||^2 + eps)
 *
 * This is the shipping L0 arithmetic. Nothing here is a stand-in: the tap
 * count, the update and the normalisation are what the headset would run.
 */
static float w_[L0_TAPS], xhist[L0_TAPS], xfhist[L0_TAPS];
static float shat[SHAT_TAPS], shist[SHAT_TAPS], yhist[SHAT_TAPS];
static int   pX = 0, pXf = 0, pS = 0, pY = 0;
static float xfPow = 1e-6f;
static float gMu   = 0.05f;
static float ancNum = 1e-9f, ancDen = 1e-9f;   /* running residual vs input */

/* Circular buffers, not shifting ones. A memmove per sample per buffer at
 * 48 kHz is ~18 M float moves/s and would dominate the very budget this
 * demo exists to prove we meet. Index 0 of the weight vector pairs with the
 * newest sample, which is what the walking read pointer below preserves. */
static inline void cpush(float *buf, int n, int *pos, float v) {
  *pos = (*pos == 0) ? n - 1 : *pos - 1;
  buf[*pos] = v;
}
/* same, but maintains the exact sum of squares of the buffer */
static inline void cpush_pow(float *buf, int n, int *pos, float v, float *pw) {
  *pos = (*pos == 0) ? n - 1 : *pos - 1;
  *pw += v*v - buf[*pos]*buf[*pos];
  if (*pw < 0) *pw = 0;                 /* float drift guard */
  buf[*pos] = v;
}
static inline float cdot(const float *a, const float *buf, int n, int pos) {
  float s = 0; int j = pos;
  for (int i = 0; i < n; i++) { s += a[i] * buf[j]; if (++j == n) j = 0; }
  return s;
}

static void l0_init() {
  memset(w_, 0, sizeof w_); memset(xhist, 0, sizeof xhist);
  memset(xfhist, 0, sizeof xfhist); memset(shist, 0, sizeof shist);
  memset(yhist, 0, sizeof yhist);
  pX = pXf = pS = pY = 0; xfPow = 0.0f; ancNum = ancDen = 1e-9f;
  /* A plausible secondary path: short delay + rolloff. On real hardware this
   * is measured once at fitting; here it is fixed so the demo is repeatable. */
  memset(shat, 0, sizeof shat);
  shat[4] = 0.80f; shat[5] = 0.45f; shat[6] = 0.20f; shat[8] = -0.10f;
}

/* one sample of FxNLMS. x = reference (noise pickup), d = disturbance at ear */
static inline float l0_step(float x, float d) {
  cpush(xhist, L0_TAPS, &pX, x);
  float y = cdot(w_, xhist, L0_TAPS, pX);          /* anti-noise            */
  cpush(yhist, SHAT_TAPS, &pY, y);
  float yS = cdot(shat, yhist, SHAT_TAPS, pY);     /* anti-noise through S  */
  float e  = d - yS;                                /* residual at the ear  */

  cpush(shist, SHAT_TAPS, &pS, x);
  float xf = cdot(shat, shist, SHAT_TAPS, pS);      /* filtered reference   */
  cpush_pow(xfhist, L0_TAPS, &pXf, xf, &xfPow);

  /* textbook NLMS: normalise by the actual energy in the filtered-reference
   * buffer. The regularisation is what keeps the very first samples, when the
   * buffer is still mostly zeros, from producing an enormous step. */
  float g = gMu * e / (xfPow + 1e-3f);
  int j = pXf;
  for (int i = 0; i < L0_TAPS; i++) {
    w_[i] += g * xfhist[j];
    if (++j == L0_TAPS) j = 0;
  }

  ancNum = 0.9995f * ancNum + 0.0005f * e * e;
  ancDen = 0.9995f * ancDen + 0.0005f * d * d;
  return e;
}

/* --------------------------------------------------------- tiny real FFT */
/* Self-contained radix-2. No external DSP dependency, so this compiles with
 * a stock Arduino ESP32 core. esp-dsp's assembly FFT is the deployment path. */
static float fre[L1_NFFT], fim[L1_NFFT];
static void fft(float *re, float *im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t;
      t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cr = 1, ci = 0;
      for (int k = 0; k < len / 2; k++) {
        float ur = re[i+k],        ui = im[i+k];
        float vr = re[i+k+len/2]*cr - im[i+k+len/2]*ci;
        float vi = re[i+k+len/2]*ci + im[i+k+len/2]*cr;
        re[i+k] = ur+vr; im[i+k] = ui+vi;
        re[i+k+len/2] = ur-vr; im[i+k+len/2] = ui-vi;
        float ncr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = ncr;
      }
    }
  }
}

/* ------------------------------------------------------- L1: enhancement */
/* Spectral gain with a tracked noise floor, then the same alpha exponent that
 * was validated offline (alpha 1.6 beat 1.0 on both PESQ and noise floor).
 * The full GTCRNLite runs on the host today; its on-chip port is partial.
 * This is a DSP enhancer, and it is labelled as one -- see docs/09.
 */
static float nfloor[L1_NFFT/2+1];      /* noise PSD estimate               */
static float psm[L1_NFFT/2+1];         /* smoothed periodogram             */
static float nmin[L1_NFFT/2+1];        /* running minimum, current window  */
static float ntmp[L1_NFFT/2+1];        /* running minimum, next window     */
static uint32_t l1Frames = 0, l1MinCnt = 0;
#define L1_MINWIN   60                 /* ~1 s of 16 ms frames             */
#define L1_MINBIAS  2.5f               /* min-of-smoothed underestimates    */
static float ovl[L1_NFFT];
static float win[L1_NFFT];
static bool  l1Ready = false;

static void l1_init() {
  for (int i = 0; i < L1_NFFT; i++)
    win[i] = sqrtf(0.5f - 0.5f * cosf(2.0f*(float)M_PI*i/L1_NFFT));
  for (int i = 0; i <= L1_NFFT/2; i++) {
    nfloor[i] = 0.0f; psm[i] = 0.0f; nmin[i] = 1e30f; ntmp[i] = 1e30f;
  }
  l1Frames = 0; l1MinCnt = 0;
  memset(ovl, 0, sizeof ovl);
  l1Ready = true;
}

/* in: L1_NFFT samples (windowed frame). out: L1_HOP enhanced samples. */
static void l1_frame(const float *in, float *out, bool removeVoices) {
  for (int i = 0; i < L1_NFFT; i++) { fre[i] = in[i]*win[i]; fim[i] = 0; }
  fft(fre, fim, L1_NFFT);

  float frameE = 0;
  for (int k = 0; k <= L1_NFFT/2; k++) {
    float m2 = fre[k]*fre[k] + fim[k]*fim[k];
    frameE += m2;
    /* Minimum statistics (Martin 2001), simplified. The periodogram is
     * smoothed FIRST -- an unsmoothed bin is chi-square with 2 dof and its
     * running minimum collapses toward zero, which is exactly the bug this
     * replaced. The minimum is then taken over a SLIDING window, so the
     * estimate can recover when the noise rises; an open-ended ratchet
     * cannot. L1_MINBIAS corrects the known downward bias of a minimum. */
    psm[k] = (l1Frames == 0) ? m2 : 0.8f*psm[k] + 0.2f*m2;
    if (psm[k] < nmin[k]) nmin[k] = psm[k];
    if (psm[k] < ntmp[k]) ntmp[k] = psm[k];
  }
  float frameDb = 10.0f*log10f(frameE/(L1_NFFT/2+1) + 1e-20f);

  /* RADIO path additionally gates whole frames that are not the wearer.
   * With one microphone the only available cue is level: the wearer's mouth
   * is ~5 cm from the boom, everyone else is metres away, so the wearer is
   * tens of dB louder. With the second mic this becomes the near-field ILD
   * test, which is a far stronger discriminator (see docs/08 section 6). */
  float frameGate = 1.0f;
  if (removeVoices && frameDb < gGateDb) frameGate = 0.02f;

  for (int k = 0; k <= L1_NFFT/2; k++) {
    float m2 = fre[k]*fre[k] + fim[k]*fim[k];
    float nf = L1_MINBIAS * (l1Frames < L1_MINWIN ? psm[k] : nfloor[k]);
    float g  = m2 / (m2 + nf + 1e-20f);              /* Wiener-ish gain      */
    g = powf(g, gAlpha);                             /* validated sharpening */
    if (g < 0.01f) g = 0.01f;                        /* floor: no dead bins  */
    g *= frameGate;
    fre[k] *= g; fim[k] *= g;
    if (k > 0 && k < L1_NFFT/2) { fre[L1_NFFT-k] =  fre[k];
                                  fim[L1_NFFT-k] = -fim[k]; }
  }
  fft(fre, fim, L1_NFFT);                            /* conj-symmetric inv   */
  for (int i = 0; i < L1_NFFT; i++) fre[i] /= L1_NFFT;
  for (int i = 1; i < L1_NFFT/2; i++) {              /* undo forward-as-inv  */
    float t = fre[i]; fre[i] = fre[L1_NFFT-i]; fre[L1_NFFT-i] = t;
  }
  /* roll the sliding minimum window */
  if (++l1MinCnt >= L1_MINWIN) {
    for (int k = 0; k <= L1_NFFT/2; k++) {
      nfloor[k] = nmin[k]; nmin[k] = ntmp[k]; ntmp[k] = psm[k];
    }
    l1MinCnt = 0;
  }
  l1Frames++;
  for (int i = 0; i < L1_NFFT; i++) ovl[i] += fre[i]*win[i];
  for (int i = 0; i < L1_HOP; i++) out[i] = ovl[i];
  memmove(ovl, ovl + L1_HOP, (L1_NFFT - L1_HOP)*sizeof(float));
  memset(ovl + L1_NFFT - L1_HOP, 0, L1_HOP*sizeof(float));
}


/* ---- deterministic demo noise (also used by the host test) ---------- */
static uint32_t rngs = 12345;
/* Two resonators + hiss. sinf() twice per sample at 48 kHz is ~96k
 * transcendentals/s and would show up in the L0 timing as if the canceller
 * were expensive, which would be a lie about the thing being measured. */
static float o1r = 1, o1i = 0, o2r = 1, o2i = 0, k1c, k1s, k2c, k2s;
static void osc_init() {
  float a1 = 2.0f*(float)M_PI*120.0f/FS48, a2 = 2.0f*(float)M_PI*310.0f/FS48;
  k1c = cosf(a1); k1s = sinf(a1); k2c = cosf(a2); k2s = sinf(a2);
  o1r = 1; o1i = 0; o2r = 1; o2i = 0;
}
static inline float noiseSample() {
  float n1r = o1r*k1c - o1i*k1s; o1i = o1r*k1s + o1i*k1c; o1r = n1r;
  float n2r = o2r*k2c - o2i*k2s; o2i = o2r*k2s + o2i*k2c; o2r = n2r;
  rngs = rngs*1664525u + 1013904223u;
  float hiss = ((int32_t)(rngs >> 8) / 8388608.0f) * 0.15f;
  return 0.45f*o1i + 0.30f*o2i + hiss;
}

static void osc_init_host(void){ osc_init(); }

#endif /* RHEAR_DSP_H */
