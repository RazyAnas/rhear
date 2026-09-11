/* Does the fixed x4 makeup clip? Measure it on the real recording. */
#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float gAlpha = 1.6f, gGateDb = -200.0f;
#include "rhear_dsp.h"
#define FS 16000
static float *loadwav(const char *p,int *n){FILE*f=fopen(p,"rb");char r[12];fread(r,1,12,f);
  long b=0;for(;;){char id[4];unsigned sz;fread(id,1,4,f);fread(&sz,4,1,f);
  if(!memcmp(id,"data",4)){b=sz;break;}fseek(f,sz+(sz&1),SEEK_CUR);}
  int m=b/2;short*p16=malloc(b);fread(p16,2,m,f);fclose(f);
  float*x=malloc(m*sizeof(float));for(int i=0;i<m;i++)x[i]=p16[i]/32768.0f;*n=m;return x;}
int main(void){
  int n; float *x=loadwav("/tmp/real_in.wav",&n);
  int M=(n/L1_HOP)*L1_HOP; float *y=calloc(M,sizeof(float));
  l1_init();
  for(int h=0;h<(M-L1_NFFT)/L1_HOP;h++) l1_frame(x+h*L1_HOP, y+h*L1_HOP, 0);
  /* what the firmware did: fixed x4 then clip at 0.99 */
  long clipped=0; double pk=0;
  for(int i=0;i<M;i++){ double v=y[i]*4.0; if(fabs(v)>pk) pk=fabs(v);
                        if(fabs(v)>0.99) clipped++; }
  printf("fixed x4 makeup : peak %.2f  -> %ld of %d samples hard-clipped (%.2f%%)\n",
         pk, clipped, M, 100.0*clipped/M);
  /* speech-matched makeup, what the fix does */
  const int F=320; int m=M/F; double sa=0,sb=0; int c=0;
  float *ea=malloc(m*4),*eb=malloc(m*4);
  for(int i=0;i<m;i++){double a=0,b2=0;
    for(int j=0;j<F;j++){a+=x[i*F+j]*x[i*F+j]; b2+=y[i*F+j]*y[i*F+j];}
    ea[i]=a/F; eb[i]=b2/F;}
  float *srt=malloc(m*4); memcpy(srt,ea,m*4);
  for(int i=0;i<m;i++)for(int j=i+1;j<m;j++) if(srt[j]<srt[i]){float t=srt[i];srt[i]=srt[j];srt[j]=t;}
  float thr=srt[(int)(m*0.70)];
  for(int i=0;i<m;i++) if(ea[i]>=thr){sa+=ea[i]; sb+=eb[i]; c++;}
  double g=sqrt(sa/sb);
  long c2=0; pk=0;
  for(int i=0;i<M;i++){double v=y[i]*g; if(fabs(v)>pk)pk=fabs(v); if(fabs(v)>0.99)c2++;}
  printf("speech-matched  : gain %.2fx  peak %.2f  -> %ld clipped (%.2f%%)\n",
         g, pk, c2, 100.0*c2/M);
  return 0;
}
