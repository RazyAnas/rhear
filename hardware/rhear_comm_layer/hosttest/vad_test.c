/* Does the Sohn LLR separate speech from noise on the real recording?
 * If the two distributions overlap, no threshold can gate correctly. */
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
static int cmpf(const void*a,const void*b){float x=*(const float*)a,y=*(const float*)b;
  return x<y?-1:x>y;}

int main(void){
  int n; float *x=loadwav("/tmp/real_in.wav",&n);
  if(!x){printf("need /tmp/real_in.wav\n");return 1;}
  int H=(n-L1_NFFT)/L1_HOP;
  float *llr=malloc(H*sizeof(float)), *en=malloc(H*sizeof(float)), o[L1_HOP];
  l1_init();
  for(int h=0;h<H;h++){
    l1_frame(x+h*L1_HOP,o,0);
    llr[h]=gVadLLR;
    double s=0; for(int i=0;i<L1_HOP;i++){double v=x[h*L1_HOP+i]; s+=v*v;}
    en[h]=10*log10(s/L1_HOP+1e-20);
  }
  /* ground truth from the INPUT energy: top 30% = speech, bottom 30% = not */
  float *srt=malloc(H*4); memcpy(srt,en,H*4); qsort(srt,H,4,cmpf);
  float hiT=srt[(int)(H*0.70)], loT=srt[(int)(H*0.30)];
  double ms=0,mn=0,vs=0,vn=0; int cs=0,cn=0;
  for(int h=0;h<H;h++){ if(en[h]>=hiT){ms+=llr[h];cs++;} else if(en[h]<=loT){mn+=llr[h];cn++;} }
  ms/=cs; mn/=cn;
  for(int h=0;h<H;h++){ if(en[h]>=hiT)vs+=(llr[h]-ms)*(llr[h]-ms);
                        else if(en[h]<=loT)vn+=(llr[h]-mn)*(llr[h]-mn); }
  vs/=cs; vn/=cn;
  printf("Sohn LLR on the real recording:\n");
  printf("  speech frames : mean %6.3f  sd %.3f  (n=%d)\n", ms, sqrt(vs), cs);
  printf("  quiet frames  : mean %6.3f  sd %.3f  (n=%d)\n", mn, sqrt(vn), cn);
  printf("  separation    : %.2f d'   %s\n", (ms-mn)/sqrt(0.5*(vs+vn)+1e-12),
         ((ms-mn)/sqrt(0.5*(vs+vn)+1e-12) > 1.5) ? "usable" : "TOO WEAK TO GATE ON");
  /* best achievable accuracy, and where the threshold should sit */
  float bestT=0, bestA=0;
  for(float t=-1.0f;t<6.0f;t+=0.02f){
    int ok=0,tot=0;
    for(int h=0;h<H;h++){
      if(en[h]>=hiT){ tot++; if(llr[h]>t) ok++; }
      else if(en[h]<=loT){ tot++; if(llr[h]<=t) ok++; }
    }
    float a=(float)ok/tot; if(a>bestA){bestA=a;bestT=t;}
  }
  printf("  best threshold %.2f -> %.1f%% correct\n", bestT, 100*bestA);
  printf("  (firmware uses open %.2f / close %.2f)\n", VAD_OPEN, VAD_CLOSE);
  return 0;
}
