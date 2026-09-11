/*
 * ============================================================================
 *  RHEAR + NSNet -- a neural network running LIVE ON THE ESP32-S3
 * ============================================================================
 *
 *  BE CLEAR ABOUT WHOSE NETWORK THIS IS.
 *  NSNet is Espressif's, shipped pre-trained and pre-quantised inside esp-sr.
 *  It is NOT our GTCRNLite. Ours runs on the host today; its on-chip port is
 *  staged in E03/port_c and the bidirectional GRU is the only part validated
 *  so far. Say this out loud in the demo -- "this is Espressif's network
 *  running on-device, ours runs on the host until the port lands" -- because
 *  a judge who discovers it themselves will discount everything else.
 *
 *  What it IS: a genuine neural noise suppressor, genuinely on the chip,
 *  genuinely real time, with no laptop in the loop.
 *
 *  ---------------------------------------------------------------------------
 *  BOARD SETTINGS -- this will not work without them
 *    Board            : ESP32S3 Dev Module
 *    Flash Size       : 16MB (128Mb)
 *    PSRAM            : OPI PSRAM
 *    Partition Scheme : "ESP SR 16M (3MB APP/6MB SPIFFS/3.9MB MODEL)"
 *
 *  That partition scheme is what flashes srmodels.bin to 0xC10000. Without
 *  it esp_srmodel_init("model") finds nothing, and this sketch falls back to
 *  ns_pro (Espressif's non-neural suppressor) and says so on the serial line.
 *  If the banner says NSNET you have the network; if it says NS_PRO you do
 *  not, and the partition scheme is why.
 *
 *  WIRING -- unchanged, and verified on this bench
 *    INMP441   SCK 38  WS 39  SD 40   L/R -> GND          [I2S_NUM_0]
 *    MAX98357A BCLK 15 LRC 16 DIN 17  VIN 5V  SD -> 3V3   [I2S_NUM_1]
 *
 *  COMMANDS @ 2000000 baud
 *    L  live: mic -> NSNet -> speaker, continuously
 *    A  A/B: record 5 s, play RAW then play DENOISED
 *    B  bypass: mic -> speaker, no processing (the comparison)
 *    M  level meter + mic format check
 *    +/- output level        X stop        ?  status
 * ============================================================================
 */
#include <Arduino.h>
#include <driver/i2s.h>
#include "model_path.h"          /* has its own extern "C" */
#include "esp_ns.h"               /* has its own extern "C" */
/* esp_nsn_iface.h and esp_nsn_models.h ship WITHOUT an extern "C" guard --
 * verified in the installed headers, which contain zero occurrences of it
 * while esp_ns.h and model_path.h both have one. The library exports the
 * plain C symbol (nm shows "T esp_nsnet_handle_from_name"), so compiling the
 * declaration as C++ mangles the call site and the link fails with
 *     undefined reference to `_Z26esp_nsnet_handle_from_namePc'
 * The leading _Z is the giveaway: that is a C++ mangled name being looked up
 * against a C symbol. Wrapping the includes is the fix. */
extern "C" {
#include "esp_nsn_iface.h"
#include "esp_nsn_models.h"
}
#include "esp_random.h"

#define MIC_PORT  I2S_NUM_0
#define AMP_PORT  I2S_NUM_1
#define MIC_SCK 38
#define MIC_WS  39
#define MIC_SD  40
#define AMP_BCLK 15
#define AMP_LRC  16
#define AMP_DIN  17
#define FS_MIC   16000
/* THE AMP RUNS AT THE MIC'S RATE. It used to run at 48 kHz with each sample
 * repeated three times, and zero-order-hold upsampling is not free: repeating
 * a sample convolves the signal with a 3-sample rectangle, which leaves
 * spectral images of the 0-8 kHz baseband sitting at 8-24 kHz, attenuated
 * only by that rectangle's sinc. A class-D amplifier happily reproduces them
 * and they intermodulate. Heard on a speaker that is a hard, electrical edge
 * riding on the voice -- present exactly when there IS voice, which is what
 * was reported, and which no amount of noise suppression can remove because
 * the suppressor never saw it.
 *
 * Matching the rates removes the entire problem rather than filtering it
 * afterwards. The MAX98357A supports 8-96 kHz, so 16 kHz is well inside spec.
 * Set FS_AMP back to 48000 only if 16 kHz misbehaves on a particular board;
 * the upsampling path below still works, it is simply no longer used. */
#define FS_AMP   16000
#define UPS      (FS_AMP / FS_MIC)
#define AB_SECS  5

/* 10 % is the verified clean operating point on this amplifier; well past it
 * distorts audibly. */
static float gOut = 0.10f;

/* ------------------------------------------------------- the neural engine */
static const esp_nsn_iface_t *nsIface = NULL;   /* NSNet, if the model is there */
static esp_nsn_data_t        *nsModel = NULL;
static ns_handle_t            nsPro   = NULL;   /* fallback, not neural         */
static int  nsChunk = 160;                      /* samples per process() call   */
static bool nsIsNeural = false;

static void nsInit() {
  srmodel_list_t *models = esp_srmodel_init("model");
  char *name = models ? esp_srmodel_filter(models, ESP_NSNET_PREFIX, NULL) : NULL;
  if (name) {
    nsIface = esp_nsnet_handle_from_name(name);
    if (nsIface) {
      nsModel = nsIface->create(name);
      if (nsModel) {
        nsChunk = nsIface->get_samp_chunksize(nsModel);
        nsIsNeural = true;
        Serial.printf("engine: NSNET  model '%s'  %d samples/frame (%.1f ms)"
                      "  rate %d Hz\n", name, nsChunk,
                      1000.0f * nsChunk / FS_MIC, nsIface->get_samp_rate(nsModel));
        return;
      }
    }
    Serial.printf("found model '%s' but could not create it\n", name);
  }
  /* No model partition. ns_pro is Espressif's classical suppressor -- still
   * useful, but do NOT call it a neural network in the demo. */
  nsPro = ns_pro_create(10, 2, FS_MIC);         /* 10 ms, aggressive, 16 kHz */
  nsChunk = FS_MIC / 100;                       /* 10 ms = 160 samples       */
  nsIsNeural = false;
  Serial.printf("engine: NS_PRO (classical, NOT neural) %d samples/frame\n", nsChunk);
  Serial.println(F("  -> set Partition Scheme to 'ESP SR 16M' to get NSNet"));
}

static inline void nsRun(int16_t *in, int16_t *out) {
  if (nsIsNeural) nsIface->process(nsModel, in, out);
  else            ns_process(nsPro, in, out);
}

/* ------------------------------------------------------------------- i2s */
static bool micInit() {
  i2s_config_t c = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = FS_MIC,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8, .dma_buf_len = 64,
    .use_apll = false, .tx_desc_auto_clear = false, .fixed_mclk = 0
  };
  i2s_pin_config_t p = { .bck_io_num = MIC_SCK, .ws_io_num = MIC_WS,
                         .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = MIC_SD };
  if (i2s_driver_install(MIC_PORT, &c, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(MIC_PORT, &p) != ESP_OK) return false;
  i2s_zero_dma_buffer(MIC_PORT); return true;
}
static bool ampInit() {
  i2s_config_t c = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = FS_AMP,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8, .dma_buf_len = 64,
    .use_apll = false, .tx_desc_auto_clear = true, .fixed_mclk = 0
  };
  i2s_pin_config_t p = { .bck_io_num = AMP_BCLK, .ws_io_num = AMP_LRC,
                         .data_out_num = AMP_DIN, .data_in_num = I2S_PIN_NO_CHANGE };
  if (i2s_driver_install(AMP_PORT, &c, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(AMP_PORT, &p) != ESP_OK) return false;
  i2s_zero_dma_buffer(AMP_PORT); return true;
}

/* DC blocker. MEMS microphones carry a standing DC offset, and the INMP441
 * is no exception. A frame-based suppressor sees that offset as signal, and
 * any per-frame gain change then steps the DC level at the frame boundary --
 * which is heard as a tick or crackle on every frame, 100 times a second.
 * That is "electrical noise added to the voice", and it is not the amplifier.
 * One-pole high pass at ~20 Hz: y[n] = x[n] - x[n-1] + 0.995*y[n-1]. */
static float dcX1 = 0.0f, dcY1 = 0.0f;
static double gDcSum = 0; static uint32_t gDcN = 0;
static inline int16_t dcBlock(int32_t s24) {
  float x = (float)s24;
  float y = x - dcX1 + 0.995f * dcY1;
  dcX1 = x; dcY1 = y;
  gDcSum += x; gDcN++;
  if (y >  32767.0f) y =  32767.0f;
  if (y < -32768.0f) y = -32768.0f;
  return (int16_t)lrintf(y);
}

static int micRead(int16_t *dst, int want) {
  static int32_t raw[512];
  int got = 0;
  while (got < want) {
    size_t nb = 0; int n = want - got; if (n > 512) n = 512;
    i2s_read(MIC_PORT, raw, n * sizeof(int32_t), &nb, portMAX_DELAY);
    int m = nb / sizeof(int32_t);
    for (int i = 0; i < m; i++) dst[got + i] = dcBlock(raw[i] >> 16);
    got += m; if (!m) break;
  }
  return got;
}
/* 16 kHz mono -> 48 kHz stereo. Both slots filled because SD on 3V3 selects
 * left-channel-only, so a frame in the right slot would be silence. */
static void ampPlay(const int16_t *src, int n) {
  static int16_t out[512 * UPS * 2];
  int i = 0;
  while (i < n) {
    int m = n - i; if (m > 512) m = 512;
    int k = 0;
    for (int j = 0; j < m; j++) {
      /* Scaling to 10% throws away 3.3 bits before a 16-bit DAC. Truncating
       * on top of that -- which is what an (int16_t) cast does -- leaves a
       * signal-correlated error, and signal-correlated error is distortion,
       * not noise: it is heard riding on the voice rather than behind it.
       * Round, and add TPDF dither so what is left is white and inaudible. */
      float fv = src[i + j] * gOut;
      float d = ((float)(esp_random() >> 16) - 32767.5f) / 65535.0f
              + ((float)(esp_random() >> 16) - 32767.5f) / 65535.0f;
      long q = lrintf(fv + d);
      if (q >  32767) q =  32767;
      if (q < -32768) q = -32768;
      int16_t v = (int16_t)q;
      for (int u = 0; u < UPS; u++) { out[k++] = v; out[k++] = v; }
    }
    size_t w = 0;
    i2s_write(AMP_PORT, out, k * sizeof(int16_t), &w, portMAX_DELAY);
    i += m;
  }
}
static float dbfs(const int16_t *x, int n) {
  double s = 0; for (int i = 0; i < n; i++) { double v = x[i] / 32768.0; s += v * v; }
  return 10.0f * log10f((float)(s / (n ? n : 1)) + 1e-12f);
}

/* ----------------------------------------------------------------- modes */
static void cmdLive(bool bypass) {
  Serial.printf("live %s -- any key to stop\n",
                bypass ? "BYPASS" : (nsIsNeural ? "NSNET" : "ns_pro"));
  int16_t in[512], out[512];
  uint32_t frames = 0, usTotal = 0;
  while (!Serial.available()) {
    micRead(in, nsChunk);
    if (bypass) { ampPlay(in, nsChunk); continue; }
    uint32_t t0 = micros();
    nsRun(in, out);
    usTotal += micros() - t0;
    ampPlay(out, nsChunk);
    if (++frames >= 100) {
      float budget = 1000.0f * nsChunk / FS_MIC;
      Serial.printf("%s  in %6.1f  out %6.1f dBFS | %4.0f us/frame of %.1f ms"
                    "  RTF %.3f\n", nsIsNeural ? "NSNET " : "ns_pro",
                    dbfs(in, nsChunk), dbfs(out, nsChunk),
                    (float)usTotal / frames, budget,
                    ((float)usTotal / frames / 1000.0f) / budget);
      frames = 0; usTotal = 0;
    }
  }
  while (Serial.available()) Serial.read();
  Serial.println(F("stopped"));
}

static void cmdAB() {
  const int N = (FS_MIC * AB_SECS / nsChunk) * nsChunk;
  static int16_t *rec = NULL, *den = NULL;
  if (!rec) rec = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!den) den = (int16_t *)ps_malloc(N * sizeof(int16_t));
  if (!rec || !den) { Serial.println(F("no PSRAM -- enable OPI PSRAM")); return; }

  Serial.printf("A/B: recording %d s -- TALK, then STOP so you can hear the gaps\n", AB_SECS);
  int got = 0; while (got < N) got += micRead(rec + got, nsChunk);

  uint32_t t0 = micros();
  for (int i = 0; i + nsChunk <= N; i += nsChunk) nsRun(rec + i, den + i);
  uint32_t us = micros() - t0;
  int nf = N / nsChunk;
  float budget = 1000.0f * nsChunk / FS_MIC;
  Serial.printf("A/B: %s processed %d frames in %.1f ms -> %.0f us/frame"
                " of %.1f ms, RTF %.3f\n", nsIsNeural ? "NSNET" : "ns_pro",
                nf, us / 1000.0f, (float)us / nf, budget,
                ((float)us / nf / 1000.0f) / budget);
  Serial.printf("A/B: raw %.1f dBFS   denoised %.1f dBFS\n", dbfs(rec, N), dbfs(den, N));
  if (gDcN) Serial.printf("A/B: mic DC offset was %.0f counts (%.1f%% of full scale)"
                          " -- removed before processing\n",
                          gDcSum / gDcN, 100.0 * fabs(gDcSum / gDcN) / 32768.0);
  /* how much headroom the 10%% output level is actually using */
  int16_t pk = 0; for (int i = 0; i < N; i++) { int16_t a = den[i] < 0 ? -den[i] : den[i];
                                                if (a > pk) pk = a; }
  Serial.printf("A/B: peak %d -> %d after the %.0f%% level"
                "  (%.1f bits of the DAC in use)\n", pk, (int)(pk * gOut), gOut * 100,
                log2f((float)(pk * gOut) + 1.0f));

  Serial.println(F("playing RAW"));      ampPlay(rec, N);
  delay(700);
  Serial.println(F("playing DENOISED")); ampPlay(den, N);
  Serial.println(F("done"));
}

static void cmdMeter() {
  Serial.println(F("meter 5 s -- speak, then go quiet"));
  int16_t b[512]; uint32_t lz = 0, rails = 0, tot = 0;
  static int32_t raw[512];
  uint32_t t0 = millis();
  while (millis() - t0 < 5000) {
    size_t nb = 0;
    i2s_read(MIC_PORT, raw, 256 * sizeof(int32_t), &nb, portMAX_DELAY);
    int m = nb / sizeof(int32_t);
    for (int i = 0; i < m; i++) {
      if ((raw[i] & 0xFF) == 0) lz++;
      if (raw[i] == (int32_t)0xFFFFFFFF || raw[i] == 0) rails++;
      b[i] = (int16_t)(raw[i] >> 16);
    }
    tot += m;
    float d = dbfs(b, m);
    int n = (int)((d + 60) / 3); if (n < 0) n = 0; if (n > 20) n = 20;
    char bar[21]; for (int i = 0; i < 20; i++) bar[i] = i < n ? '#' : '.'; bar[20] = 0;
    Serial.printf("[%s] %6.1f dBFS\n", bar, d);
  }
  Serial.printf("mic format: low-8-zero %.1f%%  railed %.1f%%  (want >95%% and <5%%)\n",
                100.0f * lz / tot, 100.0f * rails / tot);
}

void setup() {
  Serial.begin(2000000);
  delay(1200);
  Serial.println(F("\nRHEAR + NSNet -- neural noise suppression on the ESP32-S3"));
  Serial.printf("mic %s   amp %s\n", micInit() ? "OK" : "FAIL", ampInit() ? "OK" : "FAIL");
  Serial.printf("mic %d Hz -> amp %d Hz  (x%d, %s)\n", FS_MIC, FS_AMP, UPS,
                UPS == 1 ? "no resampling" : "zero-order hold, expect images");
  nsInit();
  Serial.printf("PSRAM free %u KB   output level %.0f%%\n",
                (unsigned)(ESP.getFreePsram() / 1024), gOut * 100);
  Serial.println(F("L live | A A/B | B bypass | M meter | +/- level | ? status"));
}

void loop() {
  if (!Serial.available()) { delay(10); return; }
  char c = Serial.read();
  if (c == '\n' || c == '\r' || c == ' ') return;
  switch (c) {
    case 'L': cmdLive(false); break;
    case 'B': cmdLive(true);  break;
    case 'A': cmdAB(); break;
    case 'M': cmdMeter(); break;
    case '+': gOut *= 1.4f; if (gOut > 0.25f) gOut = 0.25f;
              Serial.printf("level %.0f%%\n", gOut * 100); break;
    case '-': gOut /= 1.4f; Serial.printf("level %.0f%%\n", gOut * 100); break;
    case '?': Serial.printf("engine %s  chunk %d  level %.0f%%\n",
                            nsIsNeural ? "NSNET" : "ns_pro", nsChunk, gOut * 100); break;
    default: break;
  }
}
