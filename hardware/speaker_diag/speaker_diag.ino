/*
 * speaker_diag v2 -- corrected.
 *
 * v1 had a bug that invalidated its own results: it called amp.end() between
 * tests, and on this ESP32 Arduino core end() does not release the I2S
 * channel. Every begin() after the first therefore failed with
 *     E i2s_common: i2s_new_channel: no available channel found
 * and every tone test after that point was played through a half-initialised
 * channel. The noise heard in the amplitude and frequency ladders was that,
 * not the speaker. v2 opens I2S exactly once and never closes it.
 *
 * What v1 DID establish, because those tests ran on a healthy channel:
 * digital silence was quiet and idle I2S was quiet, so the 5 V rail, the
 * grounding and the amplifier board are all good. This is not an electrical
 * fault.
 *
 * >>> BOARD MUST BE "ESP32S3 Dev Module" <<<
 * The serial log showed "ESP32-S3-Box". That board definition targets a
 * device with its own onboard codec and its own I2S pin assignments, which
 * can conflict with GPIO 15/16/17. Change it in Tools > Board before reading
 * anything into these results.
 *
 *  GPIO15->BCLK  GPIO16->LRC  GPIO17->DIN  5V->VIN  GND->GND  3V3->SD
 */
#include <ESP_I2S.h>
#define BCLK 15
#define LRC  16
#define DIN  17
#define FS   16000
I2SClass amp(I2S_NUM_0);
static int16_t buf[512];
static bool ok = false;

static void push(int n) { if (ok) amp.write((uint8_t *)buf, n * 2 * sizeof(int16_t)); }

static void tone_(float hz, float amp01, int ms) {
  static double ph = 0;
  for (int b = 0; b < (int)((long)FS * ms / 1000 / 256); b++) {
    for (int j = 0; j < 256; j++) {
      ph += 2.0 * M_PI * hz / FS; if (ph > 2 * M_PI) ph -= 2 * M_PI;
      int16_t o = (int16_t)(sin(ph) * amp01 * 32000.0);
      buf[2*j] = o; buf[2*j+1] = o;
    }
    push(256);
  }
}
static void silence(int ms) {
  memset(buf, 0, sizeof buf);
  for (int b = 0; b < (int)((long)FS * ms / 1000 / 256); b++) push(256);
}

void setup() {
  Serial.begin(115200);
  delay(700);
  Serial.println(F("\n============== speaker_diag v2 =============="));
  Serial.println(F("Board MUST be 'ESP32S3 Dev Module', not ESP32-S3-Box."));
  amp.setPins(BCLK, LRC, DIN, -1, -1);
  ok = amp.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO);
  Serial.printf("i2s begin: %s   (opened ONCE, never closed)\n", ok ? "OK" : "FAIL");
  if (!ok) Serial.println(F("FAIL here means the pins are taken -- wrong board selected."));
  Serial.println(F("============================================\n"));
}

void loop() {
  Serial.println(F("\n[1] SILENCE 4 s -- must be dead quiet"));
  silence(4000);

  Serial.println(F("[2] AMPLITUDE at 440 Hz -- say where it turns nasty"));
  const float A[] = {0.05f, 0.10f, 0.25f, 0.50f, 1.00f};
  for (int i = 0; i < 5; i++) {
    Serial.printf("      %3.0f %%\n", A[i] * 100);
    tone_(440.0f, A[i], 1500); silence(300);
  }

  Serial.println(F("[3] FREQUENCY at 25 % -- say which rattle"));
  const float Hz[] = {200, 440, 1000, 2000, 4000};
  for (int i = 0; i < 5; i++) {
    Serial.printf("      %5.0f Hz\n", Hz[i]);
    tone_(Hz[i], 0.25f, 1500); silence(300);
  }
  Serial.println(F("--- repeating ---"));
  silence(2000);
}
