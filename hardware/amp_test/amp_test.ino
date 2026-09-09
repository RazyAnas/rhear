/*
 * MAX98357A bring-up. Nothing but the amplifier -- no mic, no tasks, no DSP,
 * so anything you hear or fail to hear is the amp path and only the amp path.
 *
 *  ESP32-S3        MAX98357A
 *  GPIO 15 ------- BCLK
 *  GPIO 16 ------- LRC
 *  GPIO 17 ------- DIN
 *  GND     ------- GND
 *  5V      ------- VIN
 *  3V3     ------- SD      <-- ADD THIS WIRE. See below.
 *
 *  SD IS NOT AN OPTIONAL PIN.
 *  On the MAX98357A, SD_MODE is an analog mode select, not a logic enable:
 *      < 0.16 V          shutdown
 *      0.16 - 0.77 V     (Left + Right) / 2
 *      0.77 - 1.4 V      Right channel only
 *      > 1.4 V           Left channel only
 *  Adafruit's board fits a divider that lands ~0.45 V when SD is left open.
 *  The cheap blue breakouts usually fit only a pull-DOWN, so "open" sits at
 *  0 V, which is SHUTDOWN. That is silent no matter what the ESP32 sends.
 *  Tying SD to 3V3 selects Left channel and guarantees the amp is awake.
 *
 *  Commands:
 *    1  1 kHz sine, I2S, 16-bit stereo   (the normal case)
 *    2  1 kHz sine, I2S, 16-bit mono
 *    3  full-scale square wave -- loudest possible, use if 1 and 2 are faint
 *    4  sweep 100 Hz -> 4 kHz, catches a speaker that only passes one band
 *    5  GPIO WIGGLE: stop I2S, toggle 15/16/17 slowly as plain outputs so a
 *       multimeter reads ~1.6 V on each. Proves the ESP32 and your wiring
 *       WITHOUT involving the amplifier at all.
 *    ?  print the wiring checklist
 */
#include <ESP_I2S.h>

#define AMP_BCLK 15
#define AMP_LRC  16
#define AMP_DIN  17
#define FS       48000

I2SClass amp(I2S_NUM_0);
static bool started = false;

static bool startAmp(i2s_slot_mode_t slots) {
  if (started) { amp.end(); started = false; delay(50); }
  amp.setPins(AMP_BCLK, AMP_LRC, AMP_DIN, -1, -1);
  started = amp.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_16BIT, slots);
  Serial.printf("i2s begin(%s) -> %s\n",
                slots == I2S_SLOT_MODE_MONO ? "mono" : "stereo",
                started ? "OK" : "FAIL");
  return started;
}

static void play(int seconds, int mode) {
  static int16_t buf[512];
  uint32_t t0 = millis(); double ph = 0; size_t total = 0;
  while (millis() - t0 < (uint32_t)seconds * 1000) {
    float f = (mode == 4)
        ? 100.0f + 3900.0f * ((millis() - t0) % 4000) / 4000.0f
        : 1000.0f;
    for (int i = 0; i < 512; i++) {
      ph += 2.0 * M_PI * f / FS;
      if (ph > 2 * M_PI) ph -= 2 * M_PI;
      float v = (mode == 3) ? (sin(ph) > 0 ? 1.0f : -1.0f) : (float)sin(ph);
      buf[i] = (int16_t)(v * 26000.0f);
    }
    total += amp.write((uint8_t *)buf, sizeof(buf));
  }
  Serial.printf("wrote %u bytes over %d s -- %s\n", (unsigned)total, seconds,
                total ? "the ESP32 IS clocking data out" : "*** WROTE NOTHING ***");
}

static void wiggle() {
  if (started) { amp.end(); started = false; }
  pinMode(AMP_BCLK, OUTPUT); pinMode(AMP_LRC, OUTPUT); pinMode(AMP_DIN, OUTPUT);
  Serial.println(F("GPIO wiggle: 15/16/17 square wave at 2 Hz for 10 s."));
  Serial.println(F("Put a multimeter (DC volts) on each pin against GND."));
  Serial.println(F("  ~1.6 V average  = pin and wire are good"));
  Serial.println(F("  0 V or 3.3 V    = broken wire, wrong pin, or short"));
  for (int i = 0; i < 20; i++) {
    int v = i & 1;
    digitalWrite(AMP_BCLK, v); digitalWrite(AMP_LRC, v); digitalWrite(AMP_DIN, v);
    delay(250);
  }
  Serial.println(F("wiggle done"));
}

static void checklist() {
  Serial.println(F("\n---------------- MAX98357A checklist ----------------"));
  Serial.println(F("1. SD  -> 3V3 with a wire.  THIS IS THE USUAL FAULT."));
  Serial.println(F("     open = shutdown on most non-Adafruit breakouts."));
  Serial.println(F("2. VIN -> 5V. At 3V3 it works but is very quiet."));
  Serial.println(F("3. GND -> ESP32 GND. Must be the SAME ground."));
  Serial.println(F("4. Speaker 4-8 ohm across the two screw terminals."));
  Serial.println(F("     NOT a piezo, NOT headphones, NOT one wire to GND."));
  Serial.println(F("     Bridge output: neither terminal is ground."));
  Serial.println(F("5. Speaker test: brush it across a AA battery. It should"));
  Serial.println(F("     click. No click = the speaker is dead."));
  Serial.println(F("6. GAIN open = 9 dB. Tie GAIN to GND for 12 dB if faint."));
  Serial.println(F("-----------------------------------------------------\n"));
}

void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println(F("\nMAX98357A bring-up"));
  checklist();
  Serial.println(F("1 sine stereo | 2 sine mono | 3 square | 4 sweep | 5 gpio | ? help"));
  startAmp(I2S_SLOT_MODE_STEREO);
}

void loop() {
  if (!Serial.available()) { delay(20); return; }
  char c = Serial.read();
  switch (c) {
    case '1': startAmp(I2S_SLOT_MODE_STEREO); Serial.println(F("sine stereo 5 s")); play(5,1); break;
    case '2': startAmp(I2S_SLOT_MODE_MONO);   Serial.println(F("sine mono 5 s"));   play(5,2); break;
    case '3': startAmp(I2S_SLOT_MODE_STEREO); Serial.println(F("SQUARE 5 s (loud)"));play(5,3); break;
    case '4': startAmp(I2S_SLOT_MODE_STEREO); Serial.println(F("sweep 8 s"));       play(8,4); break;
    case '5': wiggle(); break;
    case '?': checklist(); break;
    default: break;
  }
}
