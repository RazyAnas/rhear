/*
 * RHEAR L1 communication layer -- mic in, speaker out, host in the loop.
 *
 * This is the whole comms path that does NOT depend on the ADAU1772. The codec
 * is required for L0 (the 146 us causality budget for active cancellation);
 * L1 is a 16 ms frame-rate speech enhancer with no such constraint, so it runs
 * on the ESP32-S3's own converters.
 *
 * Pins are verbatim from hardware/rhear_phase0_netlist.py (ERC-clean):
 *   INMP441   SCK->38  WS->39  SD->40      VDD 3V3, GND, L/R->GND   [I2S_NUM_1]
 *   MAX98357A BCLK->15 LRC->16 DIN->17     VIN 5V, GND, SD->3V3             [I2S_NUM_0]
 *
 * The INMP441 sends 24 bits left-justified in a 32-bit slot, so samples are
 * read 32-bit and shifted >>8 then >>8 again to int16. Reading them as 16-bit
 * is the classic "it works but sounds like noise" bug.
 *
 * Commands on USB serial at 2000000 baud:
 *   R  record 10 s of mic and stream it up as raw int16 LE   (record.py)
 *   S  duplex stream: send mic frames up, play frames sent down (host enhances)
 *   P  local passthrough: mic straight to speaker, no host
 *   T  1 kHz test tone on the speaker -- isolates the amp from the mic
 *   ?  print status
 *   X  stop whatever is running
 */
#include <ESP_I2S.h>

#define MIC_SCK 38
#define MIC_WS  39
#define MIC_SD  40
#define AMP_BCLK 15
#define AMP_LRC  16
#define AMP_DIN  17

#define FS    16000
#define HOP   256          // matches the model's 16 ms frame
#define SECS  10

I2SClass mic(I2S_NUM_1);
I2SClass amp(I2S_NUM_0);

static int32_t raw[HOP];
static int16_t pcm[HOP];
static bool ampOK = false;

static bool startMic() {
  mic.setPins(MIC_SCK, MIC_WS, -1, MIC_SD, -1);
  return mic.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
}
static bool startAmp() {
  amp.setPins(AMP_BCLK, AMP_LRC, AMP_DIN, -1, -1);
  return amp.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
}

/* one HOP of mic audio as int16; returns samples read */
static size_t readFrame() {
  size_t n = mic.readBytes((char *)raw, sizeof(raw)) / sizeof(int32_t);
  for (size_t i = 0; i < n; i++) pcm[i] = (int16_t)(raw[i] >> 16);  // 24-in-32 -> 16
  return n;
}

void setup() {
  Serial.begin(2000000);
  delay(400);
  if (!startMic()) { Serial.println("ERR mic I2S"); while (1) delay(1000); }
  ampOK = startAmp();
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();

  if (c == '?') {
    Serial.printf("RHEAR comms  fs=%d hop=%d  mic=I2S1(%d,%d,%d)  amp=I2S0(%d,%d,%d) %s\n",
                  FS, HOP, MIC_SCK, MIC_WS, MIC_SD, AMP_BCLK, AMP_LRC, AMP_DIN,
                  ampOK ? "up" : "FAILED");
    return;
  }

  if (c == 'R') {                       // record and stream up
    size_t want = (size_t)FS * SECS, got = 0;
    while (got < want) {
      size_t n = readFrame();
      if (!n) continue;
      if (got + n > want) n = want - got;
      Serial.write((uint8_t *)pcm, n * 2);
      got += n;
    }
    return;
  }

  if (c == 'T') {                       // speaker-only test tone
    if (!ampOK) { Serial.println("ERR amp"); return; }
    for (int f = 0; f < FS * 2 / HOP; f++) {
      for (int i = 0; i < HOP; i++) {
        float t = (f * HOP + i) / (float)FS;
        pcm[i] = (int16_t)(8000 * sinf(2 * PI * 1000 * t));
      }
      amp.write((uint8_t *)pcm, HOP * 2);
    }
    Serial.println("TONE_DONE");
    return;
  }

  if (c == 'P') {                       // local passthrough, no host
    if (!ampOK) { Serial.println("ERR amp"); return; }
    Serial.println("PASSTHRU");
    while (!(Serial.available() && Serial.read() == 'X')) {
      size_t n = readFrame();
      if (n) amp.write((uint8_t *)pcm, n * 2);
    }
    Serial.println("STOP");
    return;
  }

  if (c == 'S') {                       // duplex: host enhances in the loop
    Serial.println("STREAM");
    static int16_t back[HOP];
    while (true) {
      size_t n = readFrame();
      if (n) Serial.write((uint8_t *)pcm, n * 2);       // mic frame up
      // a full frame waiting below means the host returned enhanced audio
      if (Serial.available() >= (int)(HOP * 2)) {
        Serial.readBytes((char *)back, HOP * 2);
        if (ampOK) amp.write((uint8_t *)back, HOP * 2);
      } else if (Serial.available() == 1) {
        if (Serial.read() == 'X') break;
      }
    }
    Serial.println("STOP");
    return;
  }
}
