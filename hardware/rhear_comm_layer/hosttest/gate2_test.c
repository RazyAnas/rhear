#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
int main(void){
  int n; float *x=loadwav("/tmp/real_in.wav",&n); if(!x)return 1;
  int H=(n-L1_NFFT)/L1_HOP;
  l1_init();
  float o[L1_HOP]; int16_t eo[L1_HOP], raw[L1_HOP];
  double qs=0; int qn=0, closed=0, open=0;
  float *outAll=malloc(H*L1_HOP*sizeof(float));
  for(int h=0;h<H;h++){
    l1_frame(x+h*L1_HOP,o,0);
    for(int i=0;i<L1_HOP;i++){
      raw[i]=(int16_t)(x[h*L1_HOP+i]*32000);
      float v=o[i]*1.2f; if(v>0.99f)v=0.99f; if(v<-0.99f)v=-0.99f;
      eo[i]=(int16_t)(v*32000);
    }
    gateStream(eo, L1_HOP, raw);
    double s=0, sr=0;
    for(int i=0;i<L1_HOP;i++){ double a=eo[i]/32768.0, b=raw[i]/32768.0; s+=a*a; sr+=b*b; }
    float dr=10*log10(sr/L1_HOP+1e-20), de=10*log10(s/L1_HOP+1e-20);
    if(dr < -30){ qs+=s/L1_HOP; qn++; closed++; } else open++;
    for(int i=0;i<L1_HOP;i++) outAll[h*L1_HOP+i]=eo[i]/32768.0f;
  }
  printf("streaming gate, floor-relative on RAW energy:\n");
  printf("  frames where the input was quiet (< -30 dBFS): %d of %d\n", qn, H);
  if(qn) printf("  gated output in those frames averages %.1f dB  %s\n",
                10*log10(qs/qn+1e-20),
                (10*log10(qs/qn+1e-20) < -55) ? "= BLANKED" : "= still audible");
  int below=0; int F=256, m=(H*L1_HOP)/F;
  for(int i=0;i<m;i++){ double s=0;
    for(int j=0;j<F;j++){ double v=outAll[i*F+j]; s+=v*v; }
    if(10*log10(s/F+1e-20) < -65) below++; }
  printf("  %.0f%% of all output frames below -65 dB\n", 100.0*below/m);
  return 0;
}
