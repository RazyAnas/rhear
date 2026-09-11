/* Bidirectional GRU in plain C, matching PyTorch exactly.
 *
 * This is the piece psram_test.c stopped short of, and the piece most likely
 * to be subtly wrong, so it gets validated against PyTorch on its own before
 * anything else is built on it.
 *
 * PyTorch's gate order is r, z, n and its reset gate multiplies the HIDDEN
 * projection only -- n = tanh(W_in x + b_in + r * (W_hn h + b_hn)). Getting
 * that wrong (applying r to h before the matmul, as some frameworks do)
 * produces output that looks plausible and is numerically wrong, which is
 * exactly why this test exists.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float *rd(const char *p, int n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  float *b = malloc(n * sizeof(float));
  if ((int)fread(b, sizeof(float), n, f) != n) { fprintf(stderr, "short %s\n", p); exit(1); }
  fclose(f); return b;
}
static inline float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

/* one direction. x is (T,I) row-major; out is (T,H) row-major.
 * reverse!=0 walks t from T-1 down to 0 and still writes out[t]. */
static void gru_dir(const float *x, int T, int I, int H,
                    const float *Wih, const float *Whh,
                    const float *bih, const float *bhh,
                    float *out, int reverse) {
  float *h = calloc(H, sizeof(float));
  float *gi = malloc(3 * H * sizeof(float));
  float *gh = malloc(3 * H * sizeof(float));
  for (int s = 0; s < T; s++) {
    int t = reverse ? (T - 1 - s) : s;
    const float *xt = x + (size_t)t * I;
    for (int g = 0; g < 3 * H; g++) {
      float a = bih[g], b = bhh[g];
      const float *wi = Wih + (size_t)g * I;
      for (int i = 0; i < I; i++) a += wi[i] * xt[i];
      const float *wh = Whh + (size_t)g * H;
      for (int i = 0; i < H; i++) b += wh[i] * h[i];
      gi[g] = a; gh[g] = b;
    }
    for (int k = 0; k < H; k++) {
      float r = sigm(gi[k]       + gh[k]);
      float z = sigm(gi[H + k]   + gh[H + k]);
      float n = tanhf(gi[2*H + k] + r * gh[2*H + k]);   /* r on the HIDDEN term */
      h[k] = (1.0f - z) * n + z * h[k];
    }
    memcpy(out + (size_t)t * H, h, H * sizeof(float));
  }
  free(h); free(gi); free(gh);
}

int main(void) {
  const int T = 96, I = 64, H = 24;
  float *x  = rd("gru_in.bin", T * I);
  float *rf = rd("gru_out.bin", T * 2 * H);
  float *Wf = rd("gruf_weight_ih_l0.bin", 3*H*I), *Uf = rd("gruf_weight_hh_l0.bin", 3*H*H);
  float *bf = rd("gruf_bias_ih_l0.bin", 3*H),     *cf = rd("gruf_bias_hh_l0.bin", 3*H);
  float *Wr = rd("gruf_weight_ih_l0_reverse.bin", 3*H*I), *Ur = rd("gruf_weight_hh_l0_reverse.bin", 3*H*H);
  float *br = rd("gruf_bias_ih_l0_reverse.bin", 3*H),     *cr = rd("gruf_bias_hh_l0_reverse.bin", 3*H);

  float *fwd = malloc(T * H * sizeof(float)), *bwd = malloc(T * H * sizeof(float));
  gru_dir(x, T, I, H, Wf, Uf, bf, cf, fwd, 0);
  gru_dir(x, T, I, H, Wr, Ur, br, cr, bwd, 1);

  /* PyTorch concatenates [forward | backward] on the feature axis */
  double num = 0, den = 0, mx = 0;
  for (int t = 0; t < T; t++)
    for (int k = 0; k < 2 * H; k++) {
      float got = (k < H) ? fwd[t*H + k] : bwd[t*H + (k - H)];
      float want = rf[t*2*H + k];
      double d = got - want; num += d*d; den += (double)want*want;
      if (fabs(d) > mx) mx = fabs(d);
    }
  printf("bidirectional GRU vs PyTorch: rel error %.1f dB, max abs %.2e  %s\n",
         10*log10(num/den + 1e-30), mx, (mx < 1e-4) ? "PASS" : "FAIL");
  return mx < 1e-4 ? 0 : 1;
}
