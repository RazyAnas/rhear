/* Runs the EXACT C code the ESP32 runs, on a wav file, so the on-device
 * result can be compared against the host neural model honestly.
 *   clang -O2 -std=c11 -o proc_wav proc_wav.c -lm && ./proc_wav in.wav out.wav 1.6
 */
#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float gAlpha = 1.6f, gGateDb = -200.0f;
#include "rhear_dsp.h"

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: proc_wav in.wav out.wav [alpha]\n"); return 1; }
  if (argc > 3) gAlpha = atof(argv[3]);
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("in"); return 1; }
  /* Walk the RIFF chunks to find 'data'. ffmpeg emits a LIST chunk, so the
   * first 44 bytes are NOT a canonical header and copying them blindly
   * produces a file with no data chunk marker. */
  char riff[12];
  if (fread(riff, 1, 12, f) != 12) return 1;
  long bytes = 0; unsigned rate = 16000;
  for (;;) {
    char id[4]; unsigned sz;
    if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) { fprintf(stderr, "no data chunk\n"); return 1; }
    if (!memcmp(id, "fmt ", 4)) {
      unsigned char fmt[16]; fread(fmt, 1, sz < 16 ? sz : 16, f);
      rate = fmt[4] | (fmt[5]<<8) | (fmt[6]<<16) | ((unsigned)fmt[7]<<24);
      if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
    } else if (!memcmp(id, "data", 4)) { bytes = sz; break; }
    else fseek(f, sz + (sz & 1), SEEK_CUR);
  }
  int n = bytes / 2;
  short *pcm = malloc(n * 2);
  if (fread(pcm, 2, n, f) != (size_t)n) return 1;
  fclose(f);

  float *x = malloc(n * sizeof(float)), *y = calloc(n, sizeof(float));
  for (int i = 0; i < n; i++) x[i] = pcm[i] / 32768.0f;

  l1_init();
  float fout[L1_HOP];
  int hops = (n - L1_NFFT) / L1_HOP;
  for (int h = 0; h < hops; h++) l1_frame(x + h * L1_HOP, y + h * L1_HOP, 0);

  /* makeup so speech sits at the same level as the input, measured on the
   * loud frames only -- the same rule used for the offline A/B */
  int fr = 320, m = n / fr;
  double *ex = malloc(m * sizeof(double)), *ey = malloc(m * sizeof(double));
  for (int i = 0; i < m; i++) {
    double a = 0, b = 0;
    for (int j = 0; j < fr; j++) { a += x[i*fr+j]*x[i*fr+j]; b += y[i*fr+j]*y[i*fr+j]; }
    ex[i] = a / fr; ey[i] = b / fr;
  }
  double thr = 0; { double *c = malloc(m*sizeof(double)); memcpy(c, ex, m*sizeof(double));
    for (int i=0;i<m;i++) for (int j=i+1;j<m;j++) if (c[j]<c[i]) { double t=c[i];c[i]=c[j];c[j]=t; }
    thr = c[(int)(m*0.70)]; free(c); }
  double sx = 0, sy = 0; int cnt = 0;
  for (int i = 0; i < m; i++) if (ex[i] >= thr) { sx += ex[i]; sy += ey[i]; cnt++; }
  float g = (cnt && sy > 0) ? (float)sqrt(sx / sy) : 1.0f;
  for (int i = 0; i < n; i++) { float v = y[i] * g;
    if (v > 0.99f) v = 0.99f; if (v < -0.99f) v = -0.99f; pcm[i] = (short)(v * 32000); }

  FILE *o = fopen(argv[2], "wb");
  unsigned dsz = n * 2, rsz = 36 + dsz, brate = rate * 2; unsigned short one = 1, two = 2, bits = 16;
  fwrite("RIFF", 1, 4, o); fwrite(&rsz, 4, 1, o); fwrite("WAVEfmt ", 1, 8, o);
  { unsigned s16 = 16; fwrite(&s16, 4, 1, o); }
  fwrite(&one, 2, 1, o); fwrite(&one, 2, 1, o); fwrite(&rate, 4, 1, o);
  fwrite(&brate, 4, 1, o); fwrite(&two, 2, 1, o); fwrite(&bits, 2, 1, o);
  fwrite("data", 1, 4, o); fwrite(&dsz, 4, 1, o);
  fwrite(pcm, 2, n, o); fclose(o);
  fprintf(stderr, "%s -> %s  (%d samples, %d frames, makeup %.2fx, alpha %.1f)\n",
          argv[1], argv[2], n, hops, g, gAlpha);
  return 0;
}
