/*
 * speaker_voice -- plays the real RHEAR demo audio from flash. No laptop.
 *
 * This is the A/B a judge needs to hear:
 *   BEFORE : you, plus a second person talking 2 m away, plus background
 *   AFTER  : the same 8 seconds after the near-field mask, L1, and the gate
 *
 * The two clips are level-matched ON SPEECH, so the voice is equally loud in
 * both and the ONLY audible difference is what was removed. Measured before
 * baking into flash: speech -14.0 dBFS in both, background -22.8 dBFS in
 * BEFORE and -86.7 dBFS in AFTER. That is a 64 dB difference in the noise
 * floor at identical voice level.
 *
 *  ESP32-S3        MAX98357A
 *  GPIO 15 ------- BCLK       GND ------- GND
 *  GPIO 16 ------- LRC        5V  ------- VIN
 *  GPIO 17 ------- DIN        3V3 ------- SD    (required)
 *
 * STILL TOO QUIET? Do this in hardware, it is worth more than software gain:
 *   GAIN open          =  9 dB   (what you have now)
 *   GAIN -> GND        = 12 dB   (+3 dB, one wire)
 *   GAIN -> VIN via 100k = 15 dB (+6 dB, best)
 * And a 4-8 ohm 2-3 W driver is far louder than a 16 ohm 0.25 W one; the
 * amplifier can deliver several watts into 4 ohm and almost nothing into 16.
 *
 * Serial @115200:  1 BEFORE | 2 AFTER | 3 A/B loop | + louder | - quieter
 */
#include <ESP_I2S.h>
#include "demo_audio.h"

#define BCLK 15
#define LRC  16
#define DIN  17
#define FS   16000

I2SClass amp(I2S_NUM_0);
static float gGain = 3.0f;          /* software makeup, soft-limited below */

static void startAmp() {
  amp.setPins(BCLK, LRC, DIN, -1, -1);
  bool ok = amp.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO);
  Serial.printf("i2s %s\n", ok ? "OK" : "FAIL");
}

/* mu-law byte -> int16, gain, soft limit, duplicated into BOTH slots.
 * Both slots matter: SD on 3V3 selects left-channel-only, and a frame that
 * lands in the right slot is silence. */
static void playClip(const uint8_t *pcm, uint32_t len, const char *name) {
  Serial.printf(">> %s  (%.1f s)\n", name, (float)len / FS);
  static int16_t out[512];
  for (uint32_t i = 0; i < len; i += 256) {
    uint32_t m = (len - i < 256) ? (len - i) : 256;
    for (uint32_t j = 0; j < m; j++) {
      int16_t s = (int16_t)pgm_read_word(&ULAW_TBL[pgm_read_byte(&pcm[i + j])]);
      float v = (float)s / 32000.0f * gGain;
      v = tanhf(v);                              /* soft limit, never clips  */
      int16_t o = (int16_t)(v * 30000.0f);
      out[2 * j] = o; out[2 * j + 1] = o;
    }
    amp.write((uint8_t *)out, m * 2 * sizeof(int16_t));
  }
}

static void ab() {
  playClip(BEFORE_ULAW, BEFORE_LEN, "BEFORE -- two voices and background");
  delay(700);
  playClip(AFTER_ULAW,  AFTER_LEN,  "AFTER  -- one voice, silent between words");
  delay(1200);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\nRHEAR demo audio from flash"));
  Serial.printf("BEFORE %lu samples   AFTER %lu samples   %.0f KB in flash\n",
                (unsigned long)BEFORE_LEN, (unsigned long)AFTER_LEN,
                (BEFORE_LEN + AFTER_LEN) / 1024.0f);
  Serial.println(F("1 BEFORE | 2 AFTER | 3 A/B loop | + louder | - quieter"));
  startAmp();
  ab();
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') playClip(BEFORE_ULAW, BEFORE_LEN, "BEFORE");
    else if (c == '2') playClip(AFTER_ULAW, AFTER_LEN, "AFTER");
    else if (c == '3') ab();
    else if (c == '+') { gGain *= 1.4f; Serial.printf("gain %.1f\n", gGain); }
    else if (c == '-') { gGain /= 1.4f; Serial.printf("gain %.1f\n", gGain); }
    return;
  }
  ab();
}
