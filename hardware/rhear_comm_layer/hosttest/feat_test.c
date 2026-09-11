#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float gAlpha = 2.2f, gGateDb = -200.0f;
#include "rhear_dsp.h"
#define FS 16000
static float *loadwav(const char *p,int *n){FILE*f=fopen(p,"rb");if(!f)return NULL;
  char r[12];fread(r,1,12,f);long b=0;
  for(;;){char id[4];unsigned sz;if(fread(id,1,4,f)!=4||fread(&sz,4,1,f)!=1)return NULL;
    if(!memcmp(id,"data",4)){b=sz;break;}fseek(f,sz+(sz&1),SEEK_CUR);}
  int m=b/2;short*p16=malloc(b);fread(p16,2,m,f);fclose(f);
  float*x=malloc(m*sizeof(float));for(int i=0;i<m;i++)x[i]=p16[i]/32768.0f;*n=m;return x;}
static int cf(const void*a,const void*b){float x=*(const float*)a,y=*(const float*)b;return x<y?-1:x>y;}

static void score(const char *name, float *f, float *en, int H, float hiT, float loT){
  double ms=0,mn=0,vs=0,vn=0; int cs=0,cn=0;
  for(int h=0;h<H;h++){ if(en[h]>=hiT){ms+=f[h];cs++;} else if(en[h]<=loT){mn+=f[h];cn++;} }
  ms/=cs; mn/=cn;
  for(int h=0;h<H;h++){ if(en[h]>=hiT)vs+=(f[h]-ms)*(f[h]-ms); else if(en[h]<=loT)vn+=(f[h]-mn)*(f[h]-mn); }
  vs/=cs; vn/=cn;
  float d=(ms-mn)/sqrtf(0.5f*(vs+vn)+1e-12f);
  float lo=1e30f,hi=-1e30f; for(int h=0;h<H;h++){if(f[h]<lo)lo=f[h];if(f[h]>hi)hi=f[h];}
  float bT=0,bA=0;
  for(float t=lo;t<hi;t+=(hi-lo)/400){int ok=0,tot=0;
    for(int h=0;h<H;h++){ if(en[h]>=hiT){tot++;if(f[h]>t)ok++;} else if(en[h]<=loT){tot++;if(f[h]<=t)ok++;} }
    float a=(float)ok/tot; if(a>bA){bA=a;bT=t;} }
  printf("  %-28s d' %5.2f   best thr %8.3f -> %.1f%% correct\n", name, d, bT, 100*bA);
}
int main(void){
  int n; float *x=loadwav("/tmp/real_in.wav",&n); if(!x)return 1;
  int H=(n-L1_NFFT)/L1_HOP;
  float *llr=malloc(H*4),*en=malloc(H*4),*eo=malloc(H*4),*snr=malloc(H*4),o[L1_HOP];
  l1_init();
  float floorTrack=-60;
  for(int h=0;h<H;h++){
    l1_frame(x+h*L1_HOP,o,0);
    llr[h]=gVadLLR;
    double s=0,t=0;
    for(int i=0;i<L1_HOP;i++){ s+=x[h*L1_HOP+i]*x[h*L1_HOP+i]; t+=o[i]*o[i]; }
    en[h]=10*log10(s/L1_HOP+1e-20); eo[h]=10*log10(t/L1_HOP+1e-20);
    if(eo[h]<floorTrack) floorTrack+=0.30f*(eo[h]-floorTrack); else floorTrack+=0.002f*(eo[h]-floorTrack);
    snr[h]=eo[h]-floorTrack;                 /* enhanced level above its own floor */
  }
  float *srt=malloc(H*4); memcpy(srt,en,H*4); qsort(srt,H,4,cf);
  float hiT=srt[(int)(H*0.70)], loT=srt[(int)(H*0.30)];
  printf("which feature separates speech from no-speech? (real recording)\n");
  score("raw frame energy dB", en, en, H, hiT, loT);
  score("ENHANCED frame energy dB", eo, en, H, hiT, loT);
  score("enhanced above own floor", snr, en, H, hiT, loT);
  score("Sohn LLR", llr, en, H, hiT, loT);
  return 0;
}
