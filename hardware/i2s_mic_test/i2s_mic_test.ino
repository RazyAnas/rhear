/*
 * RHEAR mic check v3 -- validates the DATA FORMAT, not just the level.
 *
 * v2 said "mic working" on a floating pin. RMS cannot tell a microphone from a
 * rail-to-rail square wave, and a floating input produces both. The reliable
 * test is structural: the INMP441 puts 24 bits left-justified in a 32-bit slot,
 * so the LOW 8 BITS OF EVERY WORD MUST BE ZERO. Noise fails that instantly.
 */
#include <ESP_I2S.h>
#define PIN_SCK 38
#define PIN_WS  39
#define PIN_SD  40
#define FS      16000
#define NS      512
I2SClass i2s;
static int32_t buf[NS];

void setup(){
  Serial.begin(115200); delay(1200);
  Serial.println("\n=== RHEAR MIC CHECK v3 (format-validated) ===");
  Serial.printf("SCK=%d WS=%d SD=%d\n", PIN_SCK, PIN_WS, PIN_SD);
  pinMode(PIN_SD, INPUT_PULLUP);   delay(20); int up=digitalRead(PIN_SD);
  pinMode(PIN_SD, INPUT_PULLDOWN); delay(20); int dn=digitalRead(PIN_SD);
  Serial.printf("SD static: pullup=%d pulldown=%d -> %s\n", up, dn,
    (up==1&&dn==0)?"FLOATING (nothing attached, or mic unpowered)":"driven");
  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD, -1);
  if(!i2s.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO)){
    Serial.printf("I2S begin failed err=%d\n", i2s.lastError()); while(1) delay(1000);
  }
  Serial.println("\n  lowbits0%  rails%   rms(dBFS)   verdict");
  Serial.println("  ------------------------------------------------------------");
}

void loop(){
  size_t n=i2s.readBytes((char*)buf,sizeof(buf))/sizeof(int32_t);
  if(!n){ Serial.println("  (no samples)"); delay(500); return; }
  int lowzero=0, rails=0; double s=0,sq=0;
  for(size_t i=0;i<n;i++){
    uint32_t w=(uint32_t)buf[i];
    if((w & 0xFF)==0) lowzero++;                       // valid left-justified word
    if(w==0xFFFFFFFF || w==0x00000000) rails++;        // stuck at a rail
    int32_t v=buf[i]>>8; s+=v; sq+=(double)v*v;
  }
  double dc=s/n, rms=sqrt(sq/n-dc*dc);
  double db= rms>0 ? 20.0*log10(rms/8388607.0) : -999.0;
  double lz=100.0*lowzero/n, rl=100.0*rails/n;
  const char *v = (lz > 95.0 && rl < 20.0) ? "VALID I2S DATA -- mic is REALLY working"
                : (rl > 40.0)              ? "RAILS -- floating pin, NOT a mic"
                : (lz < 50.0)              ? "MALFORMED -- not left-justified 24-in-32"
                                           : "marginal -- check wiring";
  Serial.printf("  %7.1f  %6.1f  %9.1f   %s\n", lz, rl, db, v);
  delay(500);
}
