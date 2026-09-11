/*
 * ============================================================================
 *  RHEAR L1 COMMUNICATION LAYER  --  built on the I2S config that actually works
 * ============================================================================
 *
 *  This replaces the ESP_I2S / I2SClass version. Everything below follows the
 *  legacy driver/i2s.h configuration that was verified working on this bench,
 *  because a proven config beats a tidier API.
 *
 *  WHAT WAS VERIFIED, AND IS THEREFORE NOT NEGOTIABLE HERE
 *    MIC  INMP441   SCK 38  WS 39  SD 40
 *         16 kHz, 32-bit slots, I2S_CHANNEL_FMT_ONLY_LEFT (L/R strapped GND),
 *         I2S_COMM_FORMAT_I2S, dma 8 x 64, sample = raw >> 16
 *    AMP  MAX98357A BCLK 15  LRC 16  DIN 17   VIN 5V   SD 3V3   GND common
 *         48 kHz, 16-bit, STEREO. Speaker straight across SPK+ / SPK-.
 *
 *  OUTPUT LEVEL IS CAPPED AT 10 %.
 *  Bench finding: ~10 % is already loud, and pushing well past it adds audible
 *  noise and distortion. The previous firmware wrote at 30000/32767 -- about
 *  91 %, roughly 9x too hot -- which is almost certainly what was heard as a
 *  buzz. OUT_LEVEL below is that operating point; '+' and '-' move it but the
 *  ceiling is deliberately 25 %.
 *
 *  The mic runs at 16 kHz (the model's rate) and the amp at 48 kHz (the rate
 *  proven on the amp), so playback repeats each sample 3x. That is a clean
 *  integer ratio, so no resampling filter is needed.
 *
 *  COMMANDS  @ 2000000 baud
 *    R  record 10 s and stream raw int16 to the host   (record.py)
 *    L  live passthrough, mic -> speaker
 *    E  live ENHANCED, mic -> L1 -> speaker
 *    A  A/B: record 5 s, play it RAW, then play it ENHANCED (on-device DSP)
 *    N  A/B with the REAL neural model -- host runs it (neural_ab.py)
 *    M  level meter, 5 s of dBFS + mic format check
 *    T  1 kHz tone -- isolates the amp
 *    + / -  output level      X stop      ?  status
 * ============================================================================
 */
#include <Arduino.h>
#include <driver/i2s.h>

/* rhear_dsp.h declares these extern VOLATILE, so the definitions must match
 * or C++ rejects the translation unit. */
volatile float gAlpha  = 1.6f;      /* L1 aggressiveness, validated offline */
volatile float gGateDb = -200.0f;   /* frame gate off by default            */
#include "rhear_dsp.h"

#define MIC_PORT  I2S_NUM_0
#define AMP_PORT  I2S_NUM_1

#define MIC_SCK 38
#define MIC_WS  39
#define MIC_SD  40
#define AMP_BCLK 15
#define AMP_LRC  16
#define AMP_DIN  17

#define FS_MIC        16000
#define FS_AMP        48000
#define UPS           (FS_AMP / FS_MIC)      /* 3 */
#define CHUNK         256                    /* 16 ms at 16 kHz = one L1 hop */
#define RECORD_SECS   10
#define AB_SECS       5

static float gOut = 0.10f;                   /* the verified operating point */

/* ------------------------------------------------------------------ setup */
static bool micInit() {
  i2s_config_t c = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = FS_MIC,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   /* L/R -> GND */
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_pin_config_t p = { .bck_io_num = MIC_SCK, .ws_io_num = MIC_WS,
                         .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = MIC_SD };
  if (i2s_driver_install(MIC_PORT, &c, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(MIC_PORT, &p) != ESP_OK) return false;
  i2s_zero_dma_buffer(MIC_PORT);
  return true;
}

static bool ampInit() {
  i2s_config_t c = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = FS_AMP,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,  /* stereo, as tested */
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };
  i2s_pin_config_t p = { .bck_io_num = AMP_BCLK, .ws_io_num = AMP_LRC,
                         .data_out_num = AMP_DIN, .data_in_num = I2S_PIN_NO_CHANGE };
  if (i2s_driver_install(AMP_PORT, &c, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(AMP_PORT, &p) != ESP_OK) return false;
  i2s_zero_dma_buffer(AMP_PORT);
  return true;
}

/* --------------------------------------------------------------------- io */
/* one CHUNK of mic audio as int16. Returns samples actually read. */
static int micRead(int16_t *dst, int want) {
  static int32_t raw[CHUNK];
  int got = 0;
  while (got < want) {
    size_t nb = 0;
    int n = want - got; if (n > CHUNK) n = CHUNK;
    i2s_read(MIC_PORT, raw, n * sizeof(int32_t), &nb, portMAX_DELAY);
    int m = nb / sizeof(int32_t);
    for (int i = 0; i < m; i++) dst[got + i] = (int16_t)(raw[i] >> 16);
    got += m;
    if (!m) break;
  }
  return got;
}

/* play int16 @16 kHz on the 48 kHz stereo amp: repeat 3x, duplicate to both
 * slots, and apply the 10 % ceiling. */
static void ampPlay(const int16_t *src, int n) {
  static int16_t out[CHUNK * UPS * 2];
  int i = 0;
  while (i < n) {
    int m = n - i; if (m > CHUNK) m = CHUNK;
    int k = 0;
    for (int j = 0; j < m; j++) {
      int16_t v = (int16_t)(src[i + j] * gOut);
      for (int u = 0; u < UPS; u++) { out[k++] = v; out[k++] = v; }
    }
    size_t wrote = 0;
    i2s_write(AMP_PORT, out, k * sizeof(int16_t), &wrote, portMAX_DELAY);
    i += m;
  }
}

static float dbfs(const int16_t *x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) { double v = x[i] / 32768.0; s += v * v; }
  return 10.0f * log10f((float)(s / (n ? n : 1)) + 1e-12f);
}

/* ----------------------------------------------------------------- modes */
static void cmdRecord() {
  Serial.println(F("REC10"));           /* host sync marker */
  delay(80);
  static int16_t buf[CHUNK];
  long total = (long)FS_MIC * RECORD_SECS, sent = 0;
  while (sent < total) {
    int n = micRead(buf, CHUNK);
    if (sent + n > total) n = total - sent;
    Serial.write((uint8_t *)buf, n * sizeof(int16_t));
    sent += n;
  }
  Serial.flush();
}

static void cmdLive(bool enhance) {
  Serial.printf("live %s -- any key to stop\n", enhance ? "ENHANCED" : "raw");
  static int16_t buf[CHUNK];
  static float win_[L1_NFFT], outf[L1_HOP];
  static int fill = 0;
  if (enhance) { l1_init(); memset(win_, 0, sizeof win_); fill = 0; }
  while (!Serial.available()) {
    int n = micRead(buf, CHUNK);
    if (!enhance) { ampPlay(buf, n); continue; }
    /* slide a 512-sample analysis window, emit one 256 hop per frame */
    memmove(win_, win_ + L1_HOP, (L1_NFFT - L1_HOP) * sizeof(float));
    for (int i = 0; i < n && i < L1_HOP; i++)
      win_[L1_NFFT - L1_HOP + i] = buf[i] / 32768.0f;
    l1_frame(win_, outf, false);
    static int16_t eo[L1_HOP];
    for (int i = 0; i < L1_HOP; i++) {
      float v = outf[i] * 4.0f;                 /* makeup for the mask */
      if (v > 0.99f) v = 0.99f; if (v < -0.99f) v = -0.99f;
      eo[i] = (int16_t)(v * 32000.0f);
    }
    ampPlay(eo, L1_HOP);
  }
  while (Serial.available()) Serial.read();
  Serial.println(F("stopped"));
}

static void cmdAB() {
  const int N = FS_MIC * AB_SECS;
  static int16_t *rec = NULL, *enh = NULL;
  if (!rec) rec = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!enh) enh = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!rec || !enh) { Serial.println(F("no PSRAM -- enable OPI PSRAM")); return; }

  Serial.printf("A/B: recording %d s -- TALK NOW\n", AB_SECS);
  int got = 0;
  while (got < N) got += micRead(rec + got, (N - got > CHUNK) ? CHUNK : N - got);
  Serial.printf("A/B: captured, %.1f dBFS. enhancing...\n", dbfs(rec, N));

  l1_init();
  memset(enh, 0, N * sizeof(int16_t));
  static float fin[L1_NFFT], fout[L1_HOP];
  int hops = (N - L1_NFFT) / L1_HOP;
  uint32_t t0 = millis();
  for (int h = 0; h < hops; h++) {
    for (int i = 0; i < L1_NFFT; i++) fin[i] = rec[h * L1_HOP + i] / 32768.0f;
    l1_frame(fin, fout, false);
    for (int i = 0; i < L1_HOP; i++) {
      float v = fout[i] * 4.0f;
      if (v > 0.99f) v = 0.99f; if (v < -0.99f) v = -0.99f;
      enh[h * L1_HOP + i] = (int16_t)(v * 32000.0f);
    }
  }
  uint32_t ms = millis() - t0;
  Serial.printf("A/B: %d frames in %lu ms = %.2f ms/frame, RTF %.3f of the 16 ms budget\n",
                hops, (unsigned long)ms, (float)ms / hops, ((float)ms / hops) / 16.0f);
  Serial.printf("A/B: raw %.1f dBFS   enhanced %.1f dBFS\n", dbfs(rec, N), dbfs(enh, N));

  Serial.println(F("A/B: playing RAW"));      ampPlay(rec, N);
  delay(600);
  Serial.println(F("A/B: playing ENHANCED")); ampPlay(enh, N);
  Serial.println(F("A/B: done"));
}


/* ---------------------------------------------------------------- neural */
/* The full GTCRNLite is NOT on this chip. Its encoder, full-band branch and
 * fusion are ported (psram_test.c) but the recurrent stage, decoder and ISTFT
 * are not, so the real model runs on the host. This mode is honest about that
 * split: the ESP32 captures and plays, the laptop does the network.
 *
 * On the device the log-MMSE path (mode A) reaches about 70 % of the neural
 * model's background reduction on real recordings -- 16.8 dB speech-to-noise
 * gap against the network's 18.9, from a raw 11.9. Use A when the laptop is
 * not in the loop, N when it is. */
static void cmdNeural() {
  const int N = FS_MIC * AB_SECS;
  static int16_t *rec = NULL, *net = NULL;
  if (!rec) rec = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!net) net = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!rec || !net) { Serial.println(F("no PSRAM -- enable OPI PSRAM")); return; }

  Serial.printf("NEURAL %d\n", N);          /* host sync marker + length */
  int got = 0;
  while (got < N) got += micRead(rec + got, (N - got > CHUNK) ? CHUNK : N - got);
  Serial.write((uint8_t *)rec, N * sizeof(int16_t));
  Serial.flush();

  /* the host writes exactly N int16 back */
  int need = N * sizeof(int16_t), have = 0;
  uint8_t *dst = (uint8_t *)net;
  uint32_t t0 = millis();
  while (have < need) {
    if (millis() - t0 > 60000) { Serial.println(F("\ntimeout waiting for host")); return; }
    int n = Serial.readBytes((char *)(dst + have), need - have);
    if (n > 0) { have += n; t0 = millis(); }
  }
  Serial.printf("\nneural returned, raw %.1f dBFS  enhanced %.1f dBFS\n",
                dbfs(rec, N), dbfs(net, N));
  Serial.println(F("playing RAW"));      ampPlay(rec, N);
  delay(600);
  Serial.println(F("playing NEURAL"));   ampPlay(net, N);
  Serial.println(F("done"));
}

static void cmdMeter() {
  Serial.println(F("meter, 5 s -- speak, tap the mic"));
  static int32_t raw[CHUNK];
  uint32_t t0 = millis(); uint32_t lz = 0, rails = 0, tot = 0;
  while (millis() - t0 < 5000) {
    size_t nb = 0;
    i2s_read(MIC_PORT, raw, sizeof(raw), &nb, portMAX_DELAY);
    int m = nb / sizeof(int32_t);
    static int16_t s16[CHUNK];
    for (int i = 0; i < m; i++) {
      if ((raw[i] & 0xFF) == 0) lz++;
      if (raw[i] == (int32_t)0xFFFFFFFF || raw[i] == 0) rails++;
      s16[i] = (int16_t)(raw[i] >> 16);
    }
    tot += m;
    float d = dbfs(s16, m);
    int bars = (int)((d + 60.0f) / 3.0f); if (bars < 0) bars = 0; if (bars > 20) bars = 20;
    char b[21]; for (int i = 0; i < 20; i++) b[i] = i < bars ? '#' : '.'; b[20] = 0;
    Serial.printf("[%s] %6.1f dBFS\n", b, d);
  }
  Serial.printf("format: low-8-zero %.1f%%  railed %.1f%%  (want >95%% and <5%%)\n",
                100.0f * lz / tot, 100.0f * rails / tot);
}

static void cmdTone() {
  Serial.println(F("1 kHz tone, 3 s, at the current output level"));
  static int16_t t[CHUNK];
  for (int b = 0; b < FS_MIC * 3 / CHUNK; b++) {
    for (int i = 0; i < CHUNK; i++)
      t[i] = (int16_t)(sinf(2.0f * (float)M_PI * 1000.0f * (b * CHUNK + i) / FS_MIC) * 32000.0f);
    ampPlay(t, CHUNK);
  }
}

void setup() {
  Serial.begin(2000000);
  delay(1200);
  Serial.println(F("\nRHEAR L1 communication layer"));
  Serial.printf("mic %s   amp %s\n", micInit() ? "OK" : "FAIL", ampInit() ? "OK" : "FAIL");
  Serial.printf("mic %d Hz -> amp %d Hz (x%d)   output level %.0f%%\n",
                FS_MIC, FS_AMP, UPS, gOut * 100);
  Serial.println(F("M meter | L live raw | E live DSP | A A/B on-device | N A/B neural via host"));
  Serial.println(F("R record10 | T tone | +/- level | ? status"));
}

void loop() {
  if (!Serial.available()) { delay(10); return; }
  char c = Serial.read();
  if (c == '\n' || c == '\r' || c == ' ') return;
  switch (c) {
    case 'R': cmdRecord(); break;
    case 'L': cmdLive(false); break;
    case 'E': cmdLive(true);  break;
    case 'A': cmdAB(); break;
    case 'N': cmdNeural(); break;
    case 'M': cmdMeter(); break;
    case 'T': cmdTone(); break;
    case '+': gOut *= 1.4f; if (gOut > 0.25f) gOut = 0.25f;
              Serial.printf("level %.0f%% (ceiling 25%%)\n", gOut * 100); break;
    case '-': gOut /= 1.4f; Serial.printf("level %.0f%%\n", gOut * 100); break;
    case '?': Serial.printf("level %.0f%%  alpha %.2f\n", gOut * 100, gAlpha); break;
    default: break;
  }
}
