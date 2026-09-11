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
#include "esp_vad.h"
#include "esp_vadn_iface.h"       /* these two DO guard themselves */
#include "esp_vadn_models.h"
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
/* neural VAD state -- declared here because gateApply() below uses it */
static const esp_vadn_iface_t *vadIface = NULL;
static model_iface_data_t     *vadModel = NULL;
static int  vadChunk = 0;
static bool vadIsNeural = false;

static float gOut = 0.10f;

/* ---- input boost --------------------------------------------------------
 * The mic peaks around 7000 of 32767 (-13 dBFS). Scaling THAT by 10% for the
 * amp leaves a peak near 700, which is 9.5 bits of a 16-bit DAC -- the log
 * said so. Six and a half bits discarded puts quantisation noise near -57 dB
 * and it is audible. Boosting the signal to use the full range BEFORE the
 * output attenuation recovers most of them: peak 28000 * 0.10 = 2800, about
 * 11.5 bits. The 10% ceiling is an amplifier limit and has to stay; wasting
 * headroom on the way to it does not. */
static float gBoost = 4.0f;

/* ---- gate: no voice, no sound ------------------------------------------
 * Missing from this sketch entirely, which is why background was still
 * audible between words no matter what the suppressor did. Threshold is
 * measured against the tracked NOISE FLOOR, not a speech level -- measured
 * d' 5.11 and 100% separation on a real recording, where a Sohn statistical
 * VAD managed only 0.97 and 77.8%. */
static float gFloor = 0.0f, gGateG = 0.0f;
static bool  gFloorInit = false, gVoiced = false;
static int   gHang = 0;
static float gVadOpen = 9.0f;   /* room bursts reached 6.3 dB over floor */
static float gVadThr  = 0.20f;      /* neural VAD detection threshold */
static uint32_t gVadSpeech = 0, gVadTotal = 0;
static bool gVadNet = false;   /* the network's arm, before the AND */
static int  gOpenRun = 0;      /* consecutive agreeing frames */
static float gLastAbove = 0.0f;
static bool  gGateOn = true;

static void gateApply(int16_t *y, int n, const int16_t *raw) {
  if (!gGateOn) { gGateG = 1.0f; return; }

  /* Neural VAD if the model loaded: it answers "is a human speaking", which
   * is the question. Energy only answers "is it loud", which is why the
   * hand-tuned gate kept holding open on background noise.
   *
   * The rates do not line up: vadnet wants 512 samples (32 ms) and the
   * suppressor runs 160 (10 ms). An earlier version tested n >= vadChunk,
   * which is never true, so the neural path was silently dead and the energy
   * gate ran instead. Accumulate into a 512-sample window, run the network
   * when it fills, and hold its decision across the frames in between. */
  if (vadIsNeural) {
    /* BOTH tests must agree before the channel opens.
     *
     * Measured on this hardware with nobody speaking: the neural VAD called
     * 96% of ambient room noise SPEECH, and sweeping its threshold from 0.20
     * to 0.50 changed that by less than 4 points -- the threshold is simply
     * not the lever. Meanwhile frame energy separated speech from silence at
     * d' 5.11 on a real recording. Neither detector is reliable alone here:
     * the network fires on room activity, and energy cannot tell a shout
     * from a slammed door. Requiring both is what makes the pair strong. */
    static int16_t vbuf[1024];
    static int vfill = 0;
    for (int i = 0; i < n; i++) {
      vbuf[vfill++] = raw[i];
      if (vfill >= vadChunk) {
        vad_state_t st = vadIface->detect(vadModel, vbuf);
        gVadNet = (st == VAD_SPEECH);
        gVadTotal++; if (gVadNet) gVadSpeech++;
        vfill = 0;
      }
    }
    /* energy arm: level against a tracked noise floor */
    double s2 = 0;
    for (int i = 0; i < n; i++) { double v = raw[i] / 32768.0; s2 += v * v; }
    float e = 10.0f * log10f((float)(s2 / n) + 1e-20f);
    if (!gFloorInit) { gFloor = e; gFloorInit = true; }
    if (e < gFloor) gFloor += 0.25f  * (e - gFloor);
    else            gFloor += 0.005f * (e - gFloor);
    gLastAbove = e - gFloor;
    /* OPEN must be EARNED, closing must be FAST.
     *
     * The diagnostic showed "OPEN no" beside "gain 1.00" for line after line:
     * the decision was right and the gain never followed it. Two reasons,
     * both here. The close ramp was 0.10 per frame, so 1.0 -> 0.002 took
     * about 59 frames (590 ms); and a SINGLE open decision armed a 12-frame
     * hangover. With isolated blips arriving every ~300 ms the gain was
     * re-armed long before it could decay, so it sat near 1.0 permanently
     * while the decision said no. A slow ramp plus a trigger-happy re-arm
     * is a gate that can never close.
     *
     * Now: two consecutive agreeing frames are required to OPEN, so a single
     * burst of room noise cannot arm anything; and the close ramp is 0.35,
     * reaching digital zero in about 15 frames (150 ms). */
    bool loud = gLastAbove > gVadOpen;
    bool want = gVoiced ? (gVadNet && gLastAbove > gVadOpen - 3.0f)
                        : (gVadNet && loud);
    if (want) { if (gOpenRun < 4) gOpenRun++; } else gOpenRun = 0;
    gVoiced = (gOpenRun >= 2);                 /* debounce the OPEN side */

    if (gVoiced) gHang = 8; else if (gHang > 0) gHang--;
    float w = (gHang > 0) ? 1.0f : 0.0f;
    gGateG += ((w > gGateG) ? 0.60f : 0.35f) * (w - gGateG);
    if (gGateG < 0.01f) gGateG = 0.0f;
    for (int i = 0; i < n; i++) y[i] = (int16_t)(y[i] * gGateG);
    return;
  }

  double s = 0;
  for (int i = 0; i < n; i++) { double v = raw[i] / 32768.0; s += v * v; }
  float e = 10.0f * log10f((float)(s / n) + 1e-20f);
  if (!gFloorInit) { gFloor = e; gFloorInit = true; }
  if (e < gFloor) gFloor += 0.25f * (e - gFloor);   /* fast down */
  else            gFloor += 0.01f * (e - gFloor);   /* slow up   */
  gLastAbove = e - gFloor;
  if (gVoiced) { if (gLastAbove < gVadOpen - 3.0f) gVoiced = false; }
  else         { if (gLastAbove > gVadOpen)        gVoiced = true;  }
  if (gVoiced) gHang = 12; else if (gHang > 0) gHang--;
  float want = (gHang > 0) ? 1.0f : 0.0f;           /* hard off: true silence */
  gGateG += ((want > gGateG) ? 0.50f : 0.10f) * (want - gGateG);
  if (gGateG < 0.002f) gGateG = 0.0f;               /* snap to digital zero */
  for (int i = 0; i < n; i++) y[i] = (int16_t)(y[i] * gGateG);
}

/* ------------------------------------------------------- the neural engine */
static const esp_nsn_iface_t *nsIface = NULL;   /* NSNet, if the model is there */
static esp_nsn_data_t        *nsModel = NULL;
static ns_handle_t            nsPro   = NULL;   /* fallback, not neural         */
static int  nsChunk = 160;                      /* samples per process() call   */
static bool nsIsNeural = false;

/* ---- neural VAD ---------------------------------------------------------
 * The Arduino core's srmodels.bin does NOT contain NSNet. Verified by
 * inspecting the image: it holds wn9_hiesp (WakeNet), mn7_en (MultiNet) and
 * vadnet1_medium, and nothing matching "nsnet". No partition setting can
 * conjure a model that is not in the bundle; getting NSNet means rebuilding
 * srmodels.bin under ESP-IDF menuconfig.
 *
 * But vadnet1_medium IS there, and it is a neural network trained for exactly
 * the question that has been the actual problem: is a human speaking right
 * now. It replaces the hand-tuned energy gate, which could never tell loud
 * noise from a voice. This is a genuine neural network, running on the chip,
 * deciding when the channel goes silent. */
static void vadInit(srmodel_list_t *models) {
  char *name = models ? esp_srmodel_filter(models, ESP_VADN_PREFIX, NULL) : NULL;
  if (!name) { Serial.println(F("VAD: no vadnet model found -- energy gate only")); return; }
  vadIface = esp_vadn_handle_from_name(name);
  if (!vadIface) { Serial.println(F("VAD: handle_from_name failed")); return; }
  /* mode, channels, min speech ms, min noise ms. 64/192 ms gives a quick
   * open and a slow close, so word onsets survive and tails are not chopped. */
  vadModel = vadIface->create(name, VAD_MODE_3, 1, 64, 192);
  if (!vadModel) { Serial.println(F("VAD: create failed")); return; }
  vadChunk = vadIface->get_samp_chunksize(vadModel);
  vadIsNeural = true;
  Serial.printf("VAD:    NEURAL '%s'  %d samples/frame (%.1f ms)  rate %d Hz\n",
                name, vadChunk, 1000.0f * vadChunk / FS_MIC,
                vadIface->get_samp_rate(vadModel));
}

static void nsInit() {
  srmodel_list_t *models = esp_srmodel_init("model");
  if (models) {
    Serial.printf("srmodels partition holds %d model(s):", models->num);
    for (int i = 0; i < models->num; i++) Serial.printf(" %s", models->model_name[i]);
    Serial.println();
  } else Serial.println(F("no 'model' partition found"));
  vadInit(models);
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
  Serial.println(F("\n---------------------------------------------------------"));
  Serial.println(F(" NSNet is NOT in the Arduino model bundle. Inspecting"));
  Serial.println(F(" srmodels.bin shows wakenet, multinet and vadnet only --"));
  Serial.println(F(" no nsnet. This is not a settings problem and no"));
  Serial.println(F(" partition scheme can fix it; the model would have to be"));
  Serial.println(F(" rebuilt under ESP-IDF menuconfig."));
  Serial.println(F(" Suppression therefore falls back to ns_pro, which is"));
  Serial.println(F(" classical, NOT neural. The VAD above IS neural and is"));
  Serial.println(F(" what decides when the channel goes silent."));
  Serial.println(F("---------------------------------------------------------\n"));
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
    for (int i = 0; i < m; i++) {
      float v = (float)(raw[i] >> 16) * gBoost;
      if (v >  32767.0f) v =  32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      dst[got + i] = dcBlock((int32_t)v);
    }
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
    gateApply(out, nsChunk, in);
    ampPlay(out, nsChunk);
    if (++frames >= 100) {
      float budget = 1000.0f * nsChunk / FS_MIC;
      Serial.printf("%s  in %6.1f  out %6.1f dBFS | %4.0f us/frame of %.1f ms"
                    "  RTF %.3f\n", nsIsNeural ? "NSNET " : "ns_pro",
                    dbfs(in, nsChunk), dbfs(out, nsChunk),
                    (float)usTotal / frames, budget,
                    ((float)usTotal / frames / 1000.0f) / budget);
      Serial.printf("        %s  gate %s   (%s VAD)\n",
                    gVoiced ? "VOICE" : " --  ",
                    gGateG > 0.5f ? "OPEN" : "SHUT",
                    vadIsNeural ? "neural" : "energy");
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

  gFloorInit = false; gVoiced = false; gGateG = 0.0f; gHang = 0;
  gVadSpeech = 0; gVadTotal = 0;
  Serial.printf("A/B: recording %d s -- TALK, then STOP so you can hear the gaps\n", AB_SECS);
  int got = 0; while (got < N) got += micRead(rec + got, nsChunk);

  uint32_t t0 = micros();
  for (int i = 0; i + nsChunk <= N; i += nsChunk) {
    nsRun(rec + i, den + i);
    gateApply(den + i, nsChunk, rec + i);
  }
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
  if (gVadTotal) {
    Serial.printf("A/B: neural VAD called %lu of %lu frames speech (%.0f%%)\n",
                  (unsigned long)gVadSpeech, (unsigned long)gVadTotal,
                  100.0f * gVadSpeech / gVadTotal);
    if (gVadSpeech == 0)
      Serial.println(F("A/B: NOTHING was called speech -- output is digital silence."
                       " If you DID talk, lower the threshold with 'v'."));
  }
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


/* ----------------------------------------------------- gate diagnosis ---- */
/* Prints what the gate is actually deciding, 31 times a second, for 12 s.
 * Every previous round of "the gate is not working" was diagnosed by ear,
 * and by ear you cannot tell "the VAD says speech constantly" from "the VAD
 * says silence constantly" from "the gate gain is not being applied". This
 * separates all three: the level, the network's verdict, and the gain that
 * actually multiplied the samples. */
static void cmdGateDiag() {
  Serial.println(F("\ngate diagnosis, 12 s. TALK for a few seconds, then GO QUIET."));
  Serial.println(F("  level  = input frame level, dBFS"));
  Serial.println(F("  VAD    = what the neural network decided"));
  Serial.println(F("  gain   = what actually multiplied the output (0.00 = silent)"));
  Serial.println(F("--------------------------------------------------------------"));
  int16_t in[512], out[512];
  gFloorInit = false; gVoiced = false; gGateG = 0.0f; gHang = 0;
  gVadSpeech = 0; gVadTotal = 0;
  uint32_t t0 = millis(); int n = 0, blanked = 0, fullyShut = 0, nAll = 0;
  while (millis() - t0 < 12000) {
    micRead(in, nsChunk);
    nsRun(in, out);
    gateApply(out, nsChunk, in);
    if (gGateG < 0.01f) fullyShut++;
    nAll++;
    if (++n % 3) continue;                    /* ~31 lines/s, readable */
    float lvl = dbfs(in, nsChunk);
    float o   = dbfs(out, nsChunk);
    if (gGateG < 0.01f) blanked++;
    int b = (int)((lvl + 60) / 3); if (b < 0) b = 0; if (b > 18) b = 18;
    char bar[19]; for (int i = 0; i < 18; i++) bar[i] = i < b ? '#' : '.'; bar[18] = 0;
    Serial.printf("[%s] %6.1f dB  net %-6s +%4.1f dB over floor  OPEN %-3s"
                  "  gain %.2f  out %6.1f dB\n",
                  bar, lvl, gVadNet ? "SPEECH" : "silent", gLastAbove,
                  gVoiced ? "yes" : "no", gGateG, o);
  }
  Serial.println(F("--------------------------------------------------------------"));
  if (gVadTotal)
    Serial.printf("neural VAD: %lu of %lu windows were SPEECH (%.0f%%)\n",
                  (unsigned long)gVadSpeech, (unsigned long)gVadTotal,
                  100.0f * gVadSpeech / gVadTotal);
  Serial.printf("gate reached digital zero on %.0f%% of ALL %d frames\n",
                100.0f * fullyShut / (nAll ? nAll : 1), nAll);
  Serial.println(F("\nHOW TO READ THIS"));
  Serial.println(F("  VAD SPEECH the whole time, even when quiet"));
  Serial.println(F("      -> network is too permissive. Press V (stricter)."));
  Serial.println(F("  VAD silent the whole time, even when you talk"));
  Serial.println(F("      -> too strict, or the mic is not being heard."));
  Serial.println(F("         Press v (easier). If the level bar never moves"));
  Serial.println(F("         when you speak, the problem is the MIC, not the gate."));
  Serial.println(F("  VAD tracks you correctly but gain never reaches 0.00"));
  Serial.println(F("      -> the gate logic is broken; tell me and I will fix it."));
  Serial.printf("  current VAD threshold %.2f   suppressor %s\n",
                gVadThr, nsIsNeural ? "NSNET" : "ns_pro");
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
  Serial.println(F("D DIAGNOSE GATE  <- run this first | L live | A A/B | B bypass | M meter"));
  Serial.println(F("N gate on/off | V/v VAD stricter/easier | I/i input boost | +/- level"));
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
    case 'D': cmdGateDiag(); break;
    case 'N': gGateOn = !gGateOn; Serial.printf("gate %s\n", gGateOn?"ON":"off"); break;
    case 'G': gVadOpen += 1.0f; Serial.printf("gate opens above %.0f dB\n", gVadOpen); break;
    case 'g': gVadOpen -= 1.0f; if (gVadOpen < 1.0f) gVadOpen = 1.0f;
              Serial.printf("gate opens above %.0f dB\n", gVadOpen); break;
    case 'V': if (vadIsNeural) { gVadThr += 0.05f;
                vadIface->set_det_threshold(vadModel, gVadThr);
                Serial.printf("VAD threshold %.2f (higher = stricter)\n", gVadThr); } break;
    case 'v': if (vadIsNeural) { gVadThr -= 0.05f; if (gVadThr < 0.05f) gVadThr = 0.05f;
                vadIface->set_det_threshold(vadModel, gVadThr);
                Serial.printf("VAD threshold %.2f (lower = opens easier)\n", gVadThr); } break;
    case 'I': gBoost *= 1.5f; Serial.printf("input boost %.1fx\n", gBoost); break;
    case 'i': gBoost /= 1.5f; Serial.printf("input boost %.1fx\n", gBoost); break;
    case '+': gOut *= 1.4f; if (gOut > 0.25f) gOut = 0.25f;
              Serial.printf("level %.0f%%\n", gOut * 100); break;
    case '-': gOut /= 1.4f; Serial.printf("level %.0f%%\n", gOut * 100); break;
    case '?': Serial.printf("suppressor %s (%s)   VAD %s   chunk %d   level %.0f%%"
                            "   gate %s\n",
                            nsIsNeural ? "NSNET" : "ns_pro",
                            nsIsNeural ? "neural" : "classical",
                            vadIsNeural ? "NEURAL vadnet1_medium" : "energy",
                            nsChunk, gOut * 100, gGateOn ? "on" : "off"); break;
    default: break;
  }
}
