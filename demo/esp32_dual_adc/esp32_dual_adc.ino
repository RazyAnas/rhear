/*
 * RHEAR Phase 0 -- Milestone 5.1b
 * Dual analog microphone capture for the on-cup coherence measurement.
 *
 * Two MAX4466 modules:
 *   mic 1 (REFERENCE) -> GPIO 1   taped to the OUTSIDE of the X3A cup shell
 *   mic 2 (ERROR)     -> GPIO 2   inside the cup, wire threaded under the cushion
 *
 * GPIO 1-10 are ADC1. Do NOT use ADC2 -- it stops working when WiFi is on.
 *
 * Differences from the snippet in docs/PHASE0-STEP-BY-STEP.md, and why:
 *
 *   1. Deadline-based pacing instead of delayMicroseconds(). The old loop's
 *      period was 125 us PLUS however long analogRead + Serial.write took,
 *      so the true sample rate was unknown and drifted. Coherence is computed
 *      against an assumed fs; a wrong or wandering fs smears the cross-spectrum
 *      and makes gamma^2 read LOW. That would look like a physics result and
 *      would actually be a timing bug.
 *
 *   2. Channel tags in the unused high bits. The ADC is 12-bit, so bits 12-15
 *      of each uint16 are free. Mic 1 samples carry 0x0, mic 2 carry 0x1. The
 *      host uses this to lock onto the stream and to detect any dropped byte
 *      instead of silently swapping the two channels for the rest of the file.
 *
 *   3. An ASCII banner before the binary, so the capture script can confirm it
 *      opened the right port and agrees about the sample rate.
 */

#define PIN_REF   1        // mic 1, outside the cup
#define PIN_ERR   2        // mic 2, at the ear
#define FS_HZ     8000     // must match coherence.py --fs
#define TAG_REF   0x0000
#define TAG_ERR   0x1000

static const uint32_t PERIOD_US = 1000000UL / FS_HZ;   // 125 us

static uint32_t next_us;
static uint32_t overruns = 0;

void setup() {
  Serial.begin(921600);
  while (!Serial) { }
  delay(300);

  analogReadResolution(12);
  analogSetPinAttenuation(PIN_REF, ADC_11db);
  analogSetPinAttenuation(PIN_ERR, ADC_11db);

  // Discard the first reads; the ADC's first conversion after config is noisy.
  for (int i = 0; i < 64; i++) { analogRead(PIN_REF); analogRead(PIN_ERR); }

  Serial.printf("RHEAR_DUAL_ADC fs=%d ref=%d err=%d\n", FS_HZ, PIN_REF, PIN_ERR);
  Serial.flush();

  next_us = micros();
}

void loop() {
  // Wait for this sample's deadline. Busy-wait: at 8 kHz the budget is 125 us
  // and there is nothing else to do on this core.
  uint32_t now = micros();
  if ((int32_t)(now - next_us) > (int32_t)PERIOD_US) {
    // We missed a whole period -- the host needs to know the capture is not
    // uniformly sampled. Resynchronise rather than accumulating skew.
    overruns++;
    next_us = now;
  }
  while ((int32_t)(micros() - next_us) < 0) { }
  next_us += PERIOD_US;

  uint16_t a = (analogRead(PIN_REF) & 0x0FFF) | TAG_REF;
  uint16_t b = (analogRead(PIN_ERR) & 0x0FFF) | TAG_ERR;

  Serial.write((uint8_t *)&a, 2);
  Serial.write((uint8_t *)&b, 2);
}
