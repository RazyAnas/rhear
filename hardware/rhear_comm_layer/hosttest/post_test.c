/* Do the gate and the line enhancer actually do what is claimed? */
#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
float gAlpha = 2.2f, gGateDb = -200.0f;
#include "rhear_dsp.h"
#include "rhear_post.h"
#define FS 16000
static float frand(void){ return (float)rand()/RAND_MAX*2.0f-1.0f; }

int main(void){
  srand(3);
  /* ---- 1. line enhancer on a 200 Hz tone buried in speech-like noise --- */
  int N = FS*4;
  float *sig = malloc(N*sizeof(float)), *tone = malloc(N*sizeof(float));
  float *sp = malloc(N*sizeof(float));
  for (int i=0;i<N;i++){
    sp[i]   = (i%16000 < 9000) ? 0.30f*sinf(2*M_PI*180*i/FS)*(0.5f+0.5f*sinf(2*M_PI*4*i/FS)) : 0.0f;
    tone[i] = 0.25f*sinf(2*M_PI*200*i/FS) + 0.12f*sinf(2*M_PI*400*i/FS);
    sig[i]  = sp[i] + tone[i] + 0.01f*frand();
  }
  aleReset();
  float *out = malloc(N*sizeof(float));
  for (int i=0;i<N;i++) out[i] = aleStep(sig[i], 0.02f);
  /* tone energy in the last second vs the input's */
  double ti=0, to=0, si=0, so=0; int c1=0,c2=0;
  for (int i=N-FS;i<N;i++){
    if (sp[i]*sp[i] < 1e-8) { ti += tone[i]*tone[i]; to += out[i]*out[i]; c1++; }
    else                    { si += sp[i]*sp[i];     so += out[i]*out[i]; c2++; }
  }
  printf("1. FxNLMS line enhancer, 200+400 Hz tone under speech:\n");
  printf("   tone-only regions : %+.1f dB of tone removed\n", 10*log10((ti/c1)/(to/c1+1e-20)));
  printf("   speech regions    : %+.1f dB level change (want near 0)\n",
         10*log10((so/c2)/(si/c2+1e-20)));

  /* ---- 2. gate on the real recording, after the mask ------------------- */
  FILE *f = fopen("/tmp/real_in.wav","rb");
  if (!f) { printf("\n2. gate: /tmp/real_in.wav missing, skipped\n"); return 0; }
  char r[12]; fread(r,1,12,f); long b=0;
  for(;;){ char id[4]; unsigned sz; fread(id,1,4,f); fread(&sz,4,1,f);
    if(!memcmp(id,"data",4)){b=sz;break;} fseek(f,sz+(sz&1),SEEK_CUR); }
  int M=b/2; short *p16=malloc(b); fread(p16,2,M,f); fclose(f);
  float *x=malloc(M*sizeof(float)), *y=calloc(M,sizeof(float));
  for(int i=0;i<M;i++) x[i]=p16[i]/32768.0f;
  l1_init();
  for(int h=0;h<(M-L1_NFFT)/L1_HOP;h++) l1_frame(x+h*L1_HOP, y+h*L1_HOP, 0);
  short *yi=malloc(M*2);
  for(int i=0;i<M;i++){ float v=y[i]; if(v>0.99f)v=0.99f; if(v<-0.99f)v=-0.99f;
                        yi[i]=(short)(v*32000); }
  int closed = gateBuffer(yi, M);
  int F=256, m=M/F, below70=0; double qs=0; int qn=0;
  for(int i=0;i<m;i++){ double s=0;
    for(int j=0;j<F;j++){ double v=yi[i*F+j]/32768.0; s+=v*v; }
    double d=10*log10(s/F+1e-20);
    if(d<-70) below70++;
    if(d<-50){ qs+=s/F; qn++; } }
  printf("\n2. gate on the real recording, after the mask:\n");
  printf("   closed %d%% of frames | %.0f%% of frames below -70 dB\n",
         closed, 100.0*below70/m);
  if (qn) printf("   those quiet frames average %.1f dB  %s\n",
                 10*log10(qs/qn+1e-20), (10*log10(qs/qn+1e-20) < -60) ? "= real silence" : "");
  return 0;
}
