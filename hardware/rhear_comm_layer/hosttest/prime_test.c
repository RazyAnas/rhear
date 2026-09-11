/* Does priming the noise tracker on ambient actually matter?
 * Build a clip that starts with speech (the bad case, what mode A did) and
 * the same clip preceded by 1.5 s of the same ambient (the primed case),
 * then compare the speech-to-background gap over the IDENTICAL speech region.
 */
#define RHEAR_HOST_TEST 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
float gAlpha = 1.6f, gGateDb = -200.0f;
#include "rhear_dsp.h"

#define FS 16000
static float *loadwav(const char *p, int *n) {
  FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  char riff[12]; fread(riff,1,12,f); long bytes=0;
  for(;;){ char id[4]; unsigned sz;
    if(fread(id,1,4,f)!=4||fread(&sz,4,1,f)!=1){fprintf(stderr,"no data\n");exit(1);}
    if(!memcmp(id,"data",4)){bytes=sz;break;} fseek(f,sz+(sz&1),SEEK_CUR); }
  int m = bytes/2; short *p16 = malloc(bytes); fread(p16,2,m,f); fclose(f);
  float *x = malloc(m*sizeof(float));
  for(int i=0;i<m;i++) x[i]=p16[i]/32768.0f;
  free(p16); *n=m; return x;
}
static void gap(const float *a, const float *b, int n, const char *tag) {
  const int F=320; int m=n/F; float *ea=malloc(m*4),*eb=malloc(m*4);
  for(int i=0;i<m;i++){double sa=0,sb=0;
    for(int j=0;j<F;j++){sa+=a[i*F+j]*a[i*F+j]; sb+=b[i*F+j]*b[i*F+j];}
    ea[i]=sa/F; eb[i]=sb/F;}
  float hi=0,lo=1e30f; for(int i=0;i<m;i++){if(ea[i]>hi)hi=ea[i]; if(ea[i]<lo)lo=ea[i];}
  float tH=lo+0.30f*(hi-lo), tL=lo+0.05f*(hi-lo);
  double as=0,ab=0,bs=0,bb=0; int cs=0,cb=0;
  for(int i=0;i<m;i++){ if(ea[i]>=tH){as+=ea[i];bs+=eb[i];cs++;}
                        if(ea[i]<=tL){ab+=ea[i];bb+=eb[i];cb++;} }
  printf("  %-26s raw gap %5.1f dB -> enhanced %5.1f dB   (%+.1f dB)\n", tag,
         10*log10((as/cs)/(ab/cb+1e-20)), 10*log10((bs/cs)/(bb/cb+1e-20)),
         10*log10((bs/cs)/(bb/cb+1e-20)) - 10*log10((as/cs)/(ab/cb+1e-20)));
  free(ea); free(eb);
}
int main(void){
  int n; float *x = loadwav("/tmp/real_in.wav", &n);
  /* find 1.5 s of the quietest material in the clip to use as ambient */
  int P = FS*3/2, best=0; double bl=1e30;
  for(int s=0;s+P<n;s+=FS/4){ double e=0; for(int i=0;i<P;i++) e+=x[s+i]*x[s+i];
    if(e<bl){bl=e;best=s;} }
  int M = FS*5; if (best+P+M > n) M = n-best-P;
  float *speech = x + (best+P < n-M ? best+P : 0);

  /* case 1: no priming -- l1_init then straight into speech (the old bug) */
  float *y1 = calloc(M,sizeof(float));
  l1_init();
  for(int h=0; h<(M-L1_NFFT)/L1_HOP; h++) l1_frame(speech+h*L1_HOP, y1+h*L1_HOP, 0);

  /* case 2: prime on the ambient, then the SAME speech */
  float *y2 = calloc(M,sizeof(float)); float tmp[L1_HOP];
  l1_init();
  for(int h=0; h<(P-L1_NFFT)/L1_HOP; h++) l1_frame(x+best+h*L1_HOP, tmp, 0);
  for(int h=0; h<(M-L1_NFFT)/L1_HOP; h++) l1_frame(speech+h*L1_HOP, y2+h*L1_HOP, 0);

  printf("priming the minimum-statistics tracker, same speech both times:\n");
  gap(speech, y1, M, "NO priming (old mode A)");
  gap(speech, y2, M, "1.5 s ambient priming");
  return 0;
}
