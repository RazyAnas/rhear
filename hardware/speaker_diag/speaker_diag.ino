/*
 * speaker_diag -- four tests that separate "the audio is wrong" from
 * "the hardware cannot do this". Runs automatically, announces each step.
 * Listen and note WHICH tests are noisy. That answer decides everything.
 *
 * TEST 1  DIGITAL SILENCE. I2S running, writing zeros.
 *         Noise here  => the fault is electrical: 5 V supply, grounding, or
 *                        the amplifier board. NOT the audio. Nothing I send
 *                        you can fix it.
 *         Quiet here  => the amp and wiring are fine.
 *
 * TEST 2  I2S STOPPED, pins released.
 *         Noise here  => power/ground only. The ESP32 is not even clocking.
 *
 * TEST 3  AMPLITUDE LADDER, 440 Hz at 5/10/25/50/100 %.
 *         Clean quiet, dirty loud => the SPEAKER is being overdriven. A
 *         16 ohm 0.25 W driver on a 5 V bridge is at its limit around 0.2 W,
 *         so full scale physically bottoms the cone out. This is the most
 *         likely answer and it is fixed by a 4-8 ohm 2-3 W driver, not code.
 *
 * TEST 4  FREQUENCY LADDER, 200/440/1k/2k/4k at 25 %.
 *         Low ones rattle, high ones clean => small cone, same conclusion.
 *
 * ALSO, WITH A MULTIMETER, while test 3 is playing loudly:
 *   measure the amp's VIN to GND. If it sags below ~4.5 V the USB rail is
 *   browning out under load, which sounds exactly like electrical noise.
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

static bool up() {
  amp.setPins(BCLK, LRC, DIN, -1, -1);
  return amp.begin(I2S_MODE_STD, FS, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO);
}
static void push(int n) { amp.write((uint8_t *)buf, n * 2 * sizeof(int16_t)); }

static void tone_(float hz, float amp01, int ms) {
  static double ph = 0;
  int blocks = (int)((long)FS * ms / 1000 / 256);
  for (int b = 0; b < blocks; b++) {
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
  Serial.println(F("\n=============== speaker_diag ==============="));
  Serial.println(F("Listen carefully and note WHICH tests are noisy."));
  Serial.println(F("============================================\n"));
}

void loop() {
  Serial.println(F("\n[TEST 1] DIGITAL SILENCE, 5 s. I2S running, writing zeros."));
  Serial.println(F("         Should be DEAD QUIET. Noise here = electrical fault."));
  up(); silence(5000);

  Serial.println(F("\n[TEST 2] I2S STOPPED, 5 s. Pins released, amp idle."));
  Serial.println(F("         Noise here = power or grounding, nothing else."));
  amp.end();
  pinMode(BCLK, INPUT); pinMode(LRC, INPUT); pinMode(DIN, INPUT);
  delay(5000);

  Serial.println(F("\n[TEST 3] AMPLITUDE LADDER at 440 Hz. Note where it turns nasty."));
  up();
  const float A[] = {0.05f, 0.10f, 0.25f, 0.50f, 1.00f};
  for (int i = 0; i < 5; i++) {
    Serial.printf("         %3.0f %% ...\n", A[i] * 100);
    tone_(440.0f, A[i], 1500); silence(400);
  }

  Serial.println(F("\n[TEST 4] FREQUENCY LADDER at 25 %. Note which rattle."));
  const float Hz[] = {200, 440, 1000, 2000, 4000};
  for (int i = 0; i < 5; i++) {
    Serial.printf("         %5.0f Hz ...\n", Hz[i]);
    tone_(Hz[i], 0.25f, 1500); silence(400);
  }

  Serial.println(F("\n--- cycle complete, repeating in 3 s ---"));
  silence(3000);
}
