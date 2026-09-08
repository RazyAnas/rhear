#include <Arduino.h>
static bool sh(int a,int b){
  pinMode(b,INPUT_PULLUP); pinMode(a,OUTPUT); digitalWrite(a,LOW); delay(8);
  bool lo=(digitalRead(b)==0);
  pinMode(b,INPUT_PULLDOWN); digitalWrite(a,HIGH); delay(8);
  bool hi=(digitalRead(b)==1);
  pinMode(a,INPUT); pinMode(b,INPUT);
  return lo&&hi;
}
void setup(){
  Serial.begin(115200); delay(1500);
  Serial.println("\n=== FOCUSED SHORT TEST: SCK38 / WS39 / SD40 ===");
  int P[3]={38,39,40};
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) if(i!=j){
    bool s=sh(P[i],P[j]);
    Serial.printf("  GPIO%d -> GPIO%d : %s\n",P[i],P[j], s?"SHORTED  ***":"clear");
  }
  Serial.println("\n--- static read of SD40 with I2S off ---");
  pinMode(40,INPUT_PULLUP);  delay(20); Serial.printf("  pullup=%d\n",digitalRead(40));
  pinMode(40,INPUT_PULLDOWN);delay(20); Serial.printf("  pulldown=%d\n",digitalRead(40));
  Serial.println("done");
}
void loop(){delay(1000);}
