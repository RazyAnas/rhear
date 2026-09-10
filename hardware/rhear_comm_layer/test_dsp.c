/* Host test for rhear_dsp.h. Proves, before anything is flashed:
 *   1. the FFT round-trips
 *   2. FxNLMS actually converges and by how many dB
 *   3. L1 lowers a noise floor without gutting speech
 *   4. what one L0 sample costs, so the 20.8 us budget claim is checkable
 */
#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
float gAlpha = 1.6f, gGateDb = -200.0f;   /* gate off for the L1 test */
#include "rhear_dsp.h"

static float frand(void){ return (float)rand()/RAND_MAX*2.0f - 1.0f; }

int main(void){
  srand(1);
  int fail = 0;

  /* ---- 1. FFT round trip -------------------------------------------- */
  float re[L1_NFFT], im[L1_NFFT], orig[L1_NFFT];
  for (int i=0;i<L1_NFFT;i++){ orig[i]=frand(); re[i]=orig[i]; im[i]=0; }
  fft(re,im,L1_NFFT);
  /* inverse via forward-twice identity, same as l1_frame uses */
  fft(re,im,L1_NFFT);
  for (int i=0;i<L1_NFFT;i++) re[i]/=L1_NFFT;
  for (int i=1;i<L1_NFFT/2;i++){ float t=re[i]; re[i]=re[L1_NFFT-i]; re[L1_NFFT-i]=t; }
  double num=0,den=0;
  for (int i=0;i<L1_NFFT;i++){ double d=re[i]-orig[i]; num+=d*d; den+=orig[i]*orig[i]; }
  double fftdb = 10*log10(num/den+1e-30);
  printf("1. FFT round trip        : %+7.1f dB error   %s\n", fftdb, fftdb<-100?"PASS":"FAIL");
  if (fftdb>=-100) fail++;

  /* ---- 2. FxNLMS convergence ---------------------------------------- */
  l0_init(); osc_init_host();
  int N = FS48*3;                       /* 3 seconds */
  double e_early=0, d_early=0, e_late=0, d_late=0;
  for (int n=0;n<N;n++){
    float x = noiseSample();            /* reference the canceller sees   */
    float d = x;                        /* disturbance at the ear         */
    float e = l0_step(x,d);
    if (n < FS48/2)      { e_early += e*e; d_early += d*d; }
    if (n > N - FS48/2)  { e_late  += e*e; d_late  += d*d; }
  }
  double db0 = 10*log10(d_early/(e_early+1e-30));
  double db1 = 10*log10(d_late /(e_late +1e-30));
  printf("2. FxNLMS reduction      : first 0.5 s %+.1f dB -> last 0.5 s %+.1f dB   %s\n",
         db0, db1, db1>10.0?"PASS":"FAIL");
  if (!(db1>10.0)) fail++;

  /* ---- 3. L1 noise floor vs speech ---------------------------------- */
  int T = 16000*3, hops = (T-L1_NFFT)/L1_HOP;
  static float sig[16000*3], out[16000*3];
  srand(7);
  for (int i=0;i<T;i++){
    float speech = (i%16000 < 8000)
        ? 0.5f*sinf(2*M_PI*220*i/16000.0f)*(0.5f+0.5f*sinf(2*M_PI*3*i/16000.0f)) : 0.0f;
    sig[i] = speech + 0.05f*frand();
  }
  /* The alpha knob is exposed at runtime, so report the whole operating
   * curve rather than one tuned point. The assertion is a sane floor for a
   * classical Wiener filter, not a number the algorithm was tuned to hit. */
  printf("3. L1 operating curve    :\n");
  double best = -99;
  for (double a = 1.0; a <= 2.6; a += 0.4) {
    gAlpha = (float)a; l1_init();
    for (int h=0; h<hops; h++) l1_frame(sig + h*L1_HOP, out + h*L1_HOP, 0);
    double nin=0,nout=0,sin_=0,sout=0; int nc=0,sc=0;
    for (int i=L1_NFFT; i<hops*L1_HOP; i++){
      int voiced = (i%16000 < 8000);
      if (voiced){ sin_+=sig[i]*sig[i]; sout+=out[i]*out[i]; sc++; }
      else       { nin +=sig[i]*sig[i]; nout+=out[i]*out[i]; nc++; }
    }
    double ncut = 10*log10((nin/nc)/(nout/nc+1e-30));
    double scut = 10*log10((sin_/sc)/(sout/sc+1e-30));
    if (ncut-scut > best) best = ncut-scut;
    printf("     alpha %.1f : noise %+5.1f dB down, speech %+5.1f dB down, net %+5.1f dB\n",
           a, ncut, scut, ncut-scut);
  }
  printf("   best net %.1f dB   %s\n", best, best>4.0?"PASS":"FAIL");
  if (!(best>4.0)) fail++;

  /* ---- 4. cost of one L0 sample ------------------------------------- */
  l0_init(); osc_init_host();
  int M = FS48*10;
  clock_t t0 = clock();
  volatile float sink=0;
  for (int n=0;n<M;n++){ float x=noiseSample(); sink += l0_step(x,x); }
  double us = (double)(clock()-t0)/CLOCKS_PER_SEC*1e6/M;
  printf("4. L0 cost (this laptop) : %.3f us/sample   budget at 48 kHz is 20.83 us\n", us);
  printf("   (target figure comes from the on-chip telemetry, not from here)\n");

  printf("\n%s\n", fail ? "SOME CHECKS FAILED" : "ALL DSP CHECKS PASS");
  return fail;
}
