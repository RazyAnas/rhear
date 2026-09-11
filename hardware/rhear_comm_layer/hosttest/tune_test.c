#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
float gAlpha = 2.2f, gGateDb = -200.0f;
#include "rhear_dsp.h"
#include "rhear_post.h"
#define FS 16000
static float *loadwav(const char *p,int *n){FILE*f=fopen(p,"rb");if(!f)return NULL;
  char r[12];fread(r,1,12,f);long b=0;
  for(;;){char id[4];unsigned sz;if(fread(id,1,4,f)!=4||fread(&sz,4,1,f)!=1)return NULL;
    if(!memcmp(id,"data",4)){b=sz;break;}fseek(f,sz+(sz&1),SEEK_CUR);}
  int m=b/2;short*p16=malloc(b);fread(p16,2,m,f);fclose(f);
  float*x=malloc(m*sizeof(float));for(int i=0;i<m;i++)x[i]=p16[i]/32768.0f;*n=m;return x;}

/* ALE with a configurable delay, so the decorrelation distance can be swept */
static float W[512], X[512+2000]; static float Pw=1.0f;
static float ale(float d,int taps,int dly,float mu){
  memmove(X+1,X,(taps+dly-1)*sizeof(float)); X[0]=d;
  const float *xd=X+dly; float y=0,p=0;
  for(int i=0;i<taps;i++){y+=W[i]*xd[i];p+=xd[i]*xd[i];}
  float e=d-y; Pw=0.99f*Pw+0.01f*p; float g=mu*e/(Pw+1e-4f);
  for(int i=0;i<taps;i++) W[i]+=g*xd[i];
  return e;
}
int main(void){
  int n; float *sp=loadwav("/tmp/real_in.wav",&n);
  if(!sp){printf("need /tmp/real_in.wav\n");return 1;}
  /* REAL speech, not a sine, plus a stationary tone */
  float *mix=malloc(n*sizeof(float)), *tone=malloc(n*sizeof(float));
  for(int i=0;i<n;i++){ tone[i]=0.20f*sinf(2*M_PI*200*i/FS)+0.10f*sinf(2*M_PI*400*i/FS);
                        mix[i]=sp[i]+tone[i]; }
  int F=320,m=n/F; float *e=malloc(m*4);
  for(int i=0;i<m;i++){double s=0;for(int j=0;j<F;j++)s+=sp[i*F+j]*sp[i*F+j];e[i]=s/F;}
  float *srt=malloc(m*4); memcpy(srt,e,m*4);
  for(int i=0;i<m;i++)for(int j=i+1;j<m;j++)if(srt[j]<srt[i]){float t=srt[i];srt[i]=srt[j];srt[j]=t;}
  float loudT=srt[(int)(m*0.75)], quietT=srt[(int)(m*0.15)];

  printf("FxNLMS line enhancer on REAL speech + a 200/400 Hz tone:\n");
  printf("  %-10s %14s %16s\n","delay","tone removed","speech damage");
  for(int dly=48; dly<=800; dly*=2){
    memset(W,0,sizeof W); memset(X,0,sizeof X); Pw=1.0f;
    float *o=malloc(n*sizeof(float));
    for(int i=0;i<n;i++) o[i]=ale(mix[i],256,dly,0.01f);
    double tq=0,oq=0,sl=0,ol=0; int cq=0,cl=0;
    for(int i=0;i<m;i++){
      double a=0,b=0,c=0;
      for(int j=0;j<F;j++){a+=tone[i*F+j]*tone[i*F+j]; b+=o[i*F+j]*o[i*F+j]; c+=sp[i*F+j]*sp[i*F+j];}
      if(e[i]<=quietT){tq+=a/F; oq+=b/F; cq++;}
      if(e[i]>=loudT ){sl+=c/F; ol+=b/F; cl++;}
    }
    printf("  %4.1f ms   %+12.1f dB %+14.1f dB\n", dly*1000.0/FS,
           10*log10((tq/cq)/(oq/cq+1e-20)), 10*log10((ol/cl)/(sl/cl+1e-20)));
    free(o);
  }
  /* gate threshold sweep on the DSP-enhanced real recording */
  int M=n; float *y=calloc(M,sizeof(float));
  l1_init();
  for(int h=0;h<(M-L1_NFFT)/L1_HOP;h++) l1_frame(sp+h*L1_HOP, y+h*L1_HOP, 0);
  printf("\ngate threshold sweep (on-device DSP output, real recording):\n");
  for(float below=20; below>=8; below-=3){
    short *yi=malloc(M*2);
    for(int i=0;i<M;i++){float v=y[i]; if(v>0.99f)v=0.99f; if(v<-0.99f)v=-0.99f; yi[i]=(short)(v*32000);}
    /* temporarily override the compiled-in threshold */
    extern int gateBufferBelow(int16_t*,int,float);
    int closed=0; { float save=below; (void)save;
      /* replicate gateBuffer with this threshold */
      int fr=GATE_FRAME, mm=M/fr; float *ee=malloc(mm*4);
      for(int i=0;i<mm;i++){double s=0;for(int j=0;j<fr;j++){double v=yi[i*fr+j]/32768.0;s+=v*v;}
        ee[i]=10*log10(s/fr+1e-20);}
      float lo=1e30f,hi=-1e30f; for(int i=0;i<mm;i++){if(ee[i]<lo)lo=ee[i];if(ee[i]>hi)hi=ee[i];}
      float spk=hi; for(int it=0;it<24;it++){float mid=0.5f*(lo+hi);int c=0;
        for(int i=0;i<mm;i++)if(ee[i]<=mid)c++; if(c*100>=mm*90){spk=mid;hi=mid;}else lo=mid;}
      float thr=spk-below; int cl2=0;
      for(int i=0;i<mm;i++){int a=i-GATE_PRE,b2=i+GATE_POST,v=0;
        if(a<0)a=0; if(b2>=mm)b2=mm-1;
        for(int j=a;j<=b2;j++) if(ee[j]>thr){v=1;break;}
        if(!v)cl2++;}
      closed=100*cl2/mm; free(ee);
    }
    printf("  %.0f dB below speech -> closes %d%% of frames\n", below, closed);
    free(yi);
  }
  return 0;
}
