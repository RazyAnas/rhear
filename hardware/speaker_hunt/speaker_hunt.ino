/*
 * speaker_hunt -- flash it and just listen. No commands, no modes.
 *
 * It plays a short melody through EIGHT different I2S configurations, one
 * after another, announcing each one over serial first. When you hear sound,
 * look at the serial monitor: the line printed just before it IS your
 * working configuration. That removes all the guessing at once.
 *
 *  ESP32-S3        MAX98357A
 *  GPIO 15 ------- BCLK
 *  GPIO 16 ------- LRC
 *  GPIO 17 ------- DIN
 *  GND     ------- GND
 *  5V      ------- VIN
 *  3V3     ------- SD     (required -- open = shutdown on most breakouts)
 *
 * If ALL EIGHT are silent, the problem is not configuration. Check, in order:
 *   - SD really on 3V3 (measure it: should read ~3.3 V, not 0 V)
 *   - VIN really on 5V, GND shared with the ESP32
 *   - speaker 4-8 ohm across BOTH screw terminals (it is a bridge output,
 *     neither terminal is ground) -- and it should click on a AA battery
 *   - board set to "ESP32S3 Dev Module", not Ozobot DRVKit
 */
#include <ESP_I2S.h>

#define BCLK 15
#define LRC  16
#define DIN  17

I2SClass amp(I2S_NUM_0);

/* a melody, so it is unmistakably "playing something" and not just hum */
static const uint16_t NOTES[] = { 523, 659, 784, 1047, 784, 659, 523, 0 };

static bool cfg(uint32_t fs, i2s_slot_mode_t slots) {
  amp.end();
  delay(60);
  amp.setPins(BCLK, LRC, DIN, -1, -1);
  return amp.begin(I2S_MODE_STD, fs, I2S_DATA_BIT_WIDTH_16BIT, slots);
}

/* writes n mono samples, duplicated into both slots when running stereo */
static void push(const int16_t *mono, size_t n, bool stereoMode) {
  static int16_t buf[512];
  if (!stereoMode) { amp.write((uint8_t *)mono, n * sizeof(int16_t)); return; }
  for (size_t i = 0; i < n && i < 256; i++) { buf[2*i] = mono[i]; buf[2*i+1] = mono[i]; }
  amp.write((uint8_t *)buf, (n < 256 ? n : 256) * 2 * sizeof(int16_t));
}

static void melody(uint32_t fs, bool stereoMode, bool square) {
  static int16_t blk[256];
  double ph = 0;
  for (int n = 0; NOTES[n] != 0; n++) {
    uint32_t need = fs * 300 / 1000;          /* 300 ms per note */
    while (need) {
      size_t m = need > 256 ? 256 : need;
      for (size_t i = 0; i < m; i++) {
        ph += 2.0 * M_PI * NOTES[n] / fs;
        if (ph > 2 * M_PI) ph -= 2 * M_PI;
        double v = square ? (sin(ph) > 0 ? 1.0 : -1.0) : sin(ph);
        blk[i] = (int16_t)(v * 26000.0);
      }
      push(blk, m, stereoMode);
      need -= m;
    }
  }
}

struct Cfg { uint32_t fs; i2s_slot_mode_t slots; bool square; const char *name; };
static const Cfg TRY[] = {
  { 48000, I2S_SLOT_MODE_STEREO, false, "48 kHz  stereo  sine"   },
  { 48000, I2S_SLOT_MODE_STEREO, true,  "48 kHz  stereo  SQUARE" },
  { 48000, I2S_SLOT_MODE_MONO,   false, "48 kHz  mono    sine"   },
  { 44100, I2S_SLOT_MODE_STEREO, false, "44.1kHz stereo  sine"   },
  { 16000, I2S_SLOT_MODE_STEREO, false, "16 kHz  stereo  sine"   },
  { 16000, I2S_SLOT_MODE_STEREO, true,  "16 kHz  stereo  SQUARE" },
  { 16000, I2S_SLOT_MODE_MONO,   false, "16 kHz  mono    sine"   },
  {  8000, I2S_SLOT_MODE_STEREO, false, "8 kHz   stereo  sine"   },
};

void setup() {
  Serial.begin(115200);
  delay(600);
  Serial.println(F("\n=========== speaker_hunt ==========="));
  Serial.println(F("Eight configurations, ~2.4 s each, then it repeats."));
  Serial.println(F("When you HEAR the melody, the line above it is your config."));
  Serial.println(F("SD must be on 3V3. VIN on 5V. Speaker across BOTH terminals."));
  Serial.println(F("====================================\n"));
}

void loop() {
  for (unsigned k = 0; k < sizeof(TRY)/sizeof(TRY[0]); k++) {
    bool ok = cfg(TRY[k].fs, TRY[k].slots);
    Serial.printf("[%u/8] %s   begin %s%s\n", k + 1, TRY[k].name,
                  ok ? "OK" : "FAIL", ok ? "  <-- listen now" : "");
    if (!ok) { delay(300); continue; }
    melody(TRY[k].fs, TRY[k].slots == I2S_SLOT_MODE_STEREO, TRY[k].square);
    delay(500);
  }
  Serial.println(F("--- all eight tried, looping ---\n"));
  delay(1200);
}
