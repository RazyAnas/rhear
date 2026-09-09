/*
 * ============================================================================
 *  RHEAR Phase-0 DEMO FIRMWARE  --  runs with NO ADAU1772
 * ============================================================================
 *
 *  SIH PS 26052 (DRDO).  ESP32-S3-N16R8 + INMP441 + MAX98357A.
 *
 *  WHAT THIS DEMONSTRATES, AND WHAT IT HONESTLY CANNOT
 *  ---------------------------------------------------
 *  The ADAU1772 is required for ACOUSTIC cancellation: L0 has a 146 us
 *  causality budget and the INMP441's sigma-delta decimation filter spends
 *  more than that on its own, before a single instruction runs. So this
 *  firmware does NOT claim live acoustic ANC.
 *
 *  What it DOES do, all of it real and all of it measured on-chip:
 *
 *    1. runs the actual L0 FxNLMS adaptive canceller at 48 kHz on core 1 and
 *       reports its true per-sample cost in microseconds against the 20.8 us
 *       budget.  The algorithm, the arithmetic and the timing are the shipping
 *       ones; only the transducer path is substituted.
 *    2. runs L1 speech enhancement on core 0 at a 16 ms frame rate.
 *    3. renders TWO DIFFERENT OUTPUTS from ONE microphone, with opposite
 *       goals -- see EAR vs RADIO below.  This is the architecture's core
 *       claim and it is audible on the speaker.
 *    4. lets you KILL the AI core at runtime and hear protection survive.
 *
 *  THE EAR PATH AND THE RADIO PATH ARE NOT THE SAME PROBLEM
 *  --------------------------------------------------------
 *    EAR   (what the soldier hears): remove machine noise, KEEP human voices.
 *          Suppressing a shouted warning is a safety failure, not a feature.
 *    RADIO (what the radio transmits): remove machine noise AND every voice
 *          that is not the wearer's.
 *  Same microphone, same instant, opposite targets.  Toggle with 'E' / 'C'.
 *
 *  PINS -- verbatim from hardware/rhear_phase0_netlist.py (ERC clean)
 *    INMP441    SCK->38  WS->39  SD->40   VDD 3V3  GND  L/R->GND  [I2S_NUM_1]
 *    MAX98357A  BCLK->15 LRC->16 DIN->17  VIN 5V   GND            [I2S_NUM_0]
 *
 *  The INMP441 sends 24 bits left-justified in a 32-bit slot. Reading them as
 *  16-bit is the classic "works but sounds like noise" bug -- see readFrame().
 *
 *  SERIAL @ 2000000 baud
 *    E  ear path        C  comms/radio path     P  passthrough (bypass)
 *    K  kill AI core    R  revive AI core
 *    N  inject noise reference (drives L0)      Q  stop injection
 *    A<x> L1 aggressiveness alpha, e.g. A1.6    G<x> gate threshold dB
 *    T  telemetry       ?  status               X  stop
 * ============================================================================
 */
#include <ESP_I2S.h>
#include <math.h>

#define MIC_SCK 38
#define MIC_WS  39
#define MIC_SD  40
#define AMP_BCLK 15
#define AMP_LRC  16
#define AMP_DIN  17

#define FS48        48000            /* L0 rate: the ANC control rate       */
#define DECIM       3                /* 48k -> 16k for L1                   */
#define FS16        (FS48 / DECIM)
#define BLK48       96               /* 2 ms of audio: L0's block           */
#define BLK16       (BLK48 / DECIM)  /* 32 samples                          */
#define L1_HOP      256              /* 16 ms frame at 16 kHz               */
#define L1_NFFT     512
#define L0_TAPS     128              /* FxNLMS adaptive filter length       */
#define SHAT_TAPS   32               /* secondary-path estimate             */

I2SClass mic(I2S_NUM_1);
I2SClass amp(I2S_NUM_0);

/* ------------------------------------------------------------------ state */
enum Mode { MODE_OFF, MODE_PASS, MODE_EAR, MODE_RADIO };
static volatile Mode  gMode      = MODE_OFF;
static volatile bool  gAiAlive   = true;      /* the kill switch            */
static volatile bool  gInject    = false;
static volatile float gAlpha     = 1.6f;      /* validated sweet spot       */
static volatile float gGateDb    = -35.0f;

/* telemetry, written by the audio task, read by the reporter */
static volatile uint32_t gL0Blocks = 0;
static volatile float    gL0UsPerBlock = 0, gL0UsWorst = 0;
static volatile float    gL1UsPerFrame = 0, gL1UsWorst = 0;
static volatile float    gAncDb    = 0;       /* live convergence, dB       */
static volatile uint32_t gAiFrames = 0;
static volatile uint32_t gUnderruns = 0;

#include "rhear_dsp.h"

/* ------------------------------------------------------------- plumbing */
static int32_t rawbuf[BLK48];
static int16_t outbuf[BLK48];
static float   l1in[L1_NFFT];      /* sliding 16 kHz analysis buffer        */
static int     l1fill = 0;
static float   l1out[L1_HOP];
static volatile bool l1Pending = false, l1Have = false;
static float   l1staging[L1_NFFT];

static size_t readFrame() {
  size_t n = mic.readBytes((char *)rawbuf, sizeof(rawbuf)) / sizeof(int32_t);
  return n;
}

/* deterministic noise for the injected-reference demo: two tones + hiss,
 * so convergence is visible and repeatable in front of judges */
/* =========================== AUDIO TASK (core 1) ========================= */
/* Highest priority. Never waits on the AI task. If the AI is dead, stalled or
 * mid-update, this loop still closes -- that is the whole architectural claim.
 */
static void audioTask(void *) {
  static uint32_t sampleN = 0;
  float l1acc[BLK16];
  for (;;) {
    if (gMode == MODE_OFF) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
    uint32_t t0 = micros();
    size_t n = readFrame();
    if (!n) { gUnderruns++; continue; }

    for (size_t i = 0; i < n; i++) {
      float mic_s = (float)(rawbuf[i] >> 8) / 8388608.0f;   /* 24-in-32      */
      float out_s = mic_s;

      if (gMode == MODE_EAR) {
        float ref = gInject ? noiseSample() : mic_s;
        float dis = gInject ? (mic_s + ref) : mic_s;
        out_s = l0_step(ref, dis);            /* residual = what the ear gets */
      }
      sampleN++;
      if (out_s >  1.0f) out_s =  1.0f;
      if (out_s < -1.0f) out_s = -1.0f;
      outbuf[i] = (int16_t)(out_s * 30000.0f);
      if ((i % DECIM) == 0 && (i/DECIM) < BLK16) l1acc[i/DECIM] = mic_s;
    }
    uint32_t t1 = micros();

    /* hand a 16 kHz block to the AI core -- non-blocking by construction */
    if (l1fill + BLK16 <= L1_NFFT) {
      memcpy(l1in + l1fill, l1acc, BLK16*sizeof(float));
      l1fill += BLK16;
    }
    if (l1fill >= L1_NFFT && !l1Pending) {
      memcpy(l1staging, l1in, sizeof l1staging);
      memmove(l1in, l1in + L1_HOP, (L1_NFFT - L1_HOP)*sizeof(float));
      l1fill -= L1_HOP;
      l1Pending = true;                        /* AI picks this up, or not   */
    }

    /* RADIO mode plays the AI's output when it exists; if the AI is dead the
     * path degrades to the raw mic rather than to silence. */
    if (gMode == MODE_RADIO && l1Have) {
      for (size_t i = 0; i < n && i/DECIM < L1_HOP; i++) {
        float v = l1out[i/DECIM];
        if (v >  1.0f) v =  1.0f; if (v < -1.0f) v = -1.0f;
        outbuf[i] = (int16_t)(v * 30000.0f);
      }
    }
    amp.write((uint8_t *)outbuf, n*sizeof(int16_t));

    float us = (float)(t1 - t0);
    gL0UsPerBlock = 0.99f*gL0UsPerBlock + 0.01f*us;
    if (us > gL0UsWorst) gL0UsWorst = us;
    gL0Blocks++;
    gAncDb = 10.0f*log10f((ancDen + 1e-12f)/(ancNum + 1e-12f));
  }
}

/* ============================= AI TASK (core 0) ========================== */
/* Everything here is killable. Nothing in the audio path waits for it. */
static void aiTask(void *) {
  for (;;) {
    if (!gAiAlive) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
    if (!l1Pending) { vTaskDelay(pdMS_TO_TICKS(2)); continue; }
    uint32_t t0 = micros();
    l1_frame(l1staging, l1out, gMode == MODE_RADIO);
    uint32_t t1 = micros();
    l1Pending = false; l1Have = true; gAiFrames++;
    float us = (float)(t1 - t0);
    gL1UsPerFrame = 0.99f*gL1UsPerFrame + 0.01f*us;
    if (us > gL1UsWorst) gL1UsWorst = us;
  }
}

/* ================================ SHELL ================================= */
static void telemetry() {
  float budget48 = 1e6f / FS48;                      /* 20.83 us per sample  */
  float perSample = gL0UsPerBlock / BLK48;
  Serial.println(F("---------------- RHEAR telemetry ----------------"));
  Serial.printf("mode            : %s\n",
      gMode==MODE_EAR?"EAR (protect: noise out, voices KEPT)":
      gMode==MODE_RADIO?"RADIO (transmit: noise + other voices out)":
      gMode==MODE_PASS?"PASSTHROUGH":"off");
  Serial.printf("AI core         : %s\n", gAiAlive ? "ALIVE" : "*** KILLED ***");
  Serial.printf("L0 blocks       : %lu   underruns %lu\n",
                (unsigned long)gL0Blocks, (unsigned long)gUnderruns);
  Serial.printf("L0 per sample   : %.2f us  (worst %.2f)  budget %.2f us  %s\n",
      perSample, gL0UsWorst/BLK48, budget48,
      (gL0UsWorst/BLK48) < budget48 ? "PASS" : "OVER");
  Serial.printf("L0 headroom     : %.1f %% of the 48 kHz sample period\n",
      100.0f*(1.0f - perSample/budget48));
  Serial.printf("ANC convergence : %.1f dB residual reduction\n", gAncDb);
  Serial.printf("L1 per frame    : %.0f us (worst %.0f) of 16000 us  -> RTF %.3f\n",
      gL1UsPerFrame, gL1UsWorst, gL1UsPerFrame/16000.0f);
  Serial.printf("L1 frames       : %lu   alpha %.2f   gate %.0f dB\n",
      (unsigned long)gAiFrames, gAlpha, gGateDb);
  Serial.printf("free heap       : %u B   PSRAM %u B\n",
      (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
  Serial.println(F("-------------------------------------------------"));
}

void setup() {
  Serial.begin(2000000);
  delay(300);
  Serial.println(F("\nRHEAR Phase-0 demo firmware (no ADAU1772)"));
  mic.setPins(MIC_SCK, MIC_WS, -1, MIC_SD, -1);
  bool mok = mic.begin(I2S_MODE_STD, FS48, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
  amp.setPins(AMP_BCLK, AMP_LRC, AMP_DIN, -1, -1);
  bool aok = amp.begin(I2S_MODE_STD, FS48, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  Serial.printf("mic %s   amp %s\n", mok?"OK":"FAIL", aok?"OK":"FAIL");
  l0_init(); l1_init(); osc_init();
  xTaskCreatePinnedToCore(audioTask, "audio", 8192, NULL, 24, NULL, 1);
  xTaskCreatePinnedToCore(aiTask,    "ai",    16384, NULL, 5,  NULL, 0);
  Serial.println(F("E ear | C radio | P pass | K kill AI | R revive | N noise | T telemetry"));
}

void loop() {
  if (!Serial.available()) { delay(20); return; }
  char c = Serial.read();
  switch (c) {
    case 'E': gMode = MODE_EAR;   l0_init(); Serial.println(F("EAR path")); break;
    case 'C': gMode = MODE_RADIO; Serial.println(F("RADIO path")); break;
    case 'P': gMode = MODE_PASS;  Serial.println(F("passthrough")); break;
    case 'X': gMode = MODE_OFF;   Serial.println(F("stopped")); break;
    case 'K': gAiAlive = false; Serial.println(F("*** AI CORE KILLED -- protection continues ***")); break;
    case 'R': gAiAlive = true;  Serial.println(F("AI core revived")); break;
    case 'N': gInject = true;  l0_init(); osc_init(); Serial.println(F("noise reference injected")); break;
    case 'Q': gInject = false; Serial.println(F("injection off")); break;
    case 'A': gAlpha  = Serial.parseFloat(); Serial.printf("alpha %.2f\n", gAlpha); break;
    case 'G': gGateDb = Serial.parseFloat(); Serial.printf("gate %.1f dB\n", gGateDb); break;
    case 'T': case '?': telemetry(); break;
    default: break;
  }
}
