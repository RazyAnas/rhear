/* =======================================================================
 * ESPRESSIF'S OWN EXAMPLE. Not written by me.
 *
 * Source, verbatim except for the three pin numbers below:
 *   ~/Library/Arduino15/packages/esp32/hardware/esp32/3.3.11/
 *       libraries/ESP_I2S/examples/Simple_tone/Simple_tone.ino
 * You can also open it yourself: File > Examples > ESP_I2S > Simple_tone
 *
 * It names the MAX98357A in its own header, so it is the reference for
 * exactly this part. Three things differ from the sketches I wrote, and any
 * of them could be the fault:
 *   - it constructs I2SClass with NO port argument (I2SClass i2s;)
 *   - it calls setPins() with THREE arguments, not five
 *   - it writes ONE 16-bit sample at a time rather than in blocks
 *
 * BOARD: "ESP32S3 Dev Module".  Your log showed ESP32-S3-Box, whose board
 * definition targets a device with its own codec and its own I2S pins.
 *
 * WIRING: BCLK->15  LRC->16  DIN->17  VIN->5V  GND->GND  SD->3V3
 *
 * If this is silent, stop testing software -- the fault is the board
 * selection, the wiring or the part. If it works, the fault was mine.
 * ======================================================================= */
/*
 This example generates a square wave based tone at a specified frequency
 and sample rate. Then outputs the data using the I2S interface to a
 MAX08357 I2S Amp Breakout board.
 I2S Circuit:
 * Arduino/Genuino Zero, MKR family and Nano 33 IoT
 * MAX08357:
   * GND connected GND
   * VIN connected 5V
   * LRC connected to pin 0 (Zero) or 3 (MKR), A2 (Nano) or 25 (ESP32)
   * BCLK connected to pin 1 (Zero) or 2 (MKR), A3 (Nano) or 5 (ESP32)
   * DIN connected to pin 9 (Zero) or A6 (MKR), 4 (Nano) or 26 (ESP32)
 DAC Circuit:
 * ESP32 or ESP32-S2
 * Audio amplifier
   - Note:
     - ESP32 has DAC on GPIO pins 25 and 26.
     - ESP32-S2 has DAC on GPIO pins 17 and 18.
  - Connect speaker(s) or headphones.
 created 17 November 2016
 by Sandeep Mistry
 For ESP extended
 Tomas Pilny
 2nd September 2021
 Lucas Saavedra Vaz (lucasssvaz)
 22nd December 2023
 anon
 10nd February 2025
 */

#include <Arduino.h>
#include <ESP_I2S.h>

// The GPIO pins are not fixed, most other pins could be used for the I2S function.
#define I2S_LRC  16   // was 25 in the stock example
#define I2S_BCLK 15   // was 5  in the stock example
#define I2S_DIN  17   // was 26 in the stock example

const int frequency = 440;    // frequency of square wave in Hz
const int amplitude = 16000;    // amplitude of square wave  <-- ~ -36 dBFS, deliberately quiet.
                              // Raise to 8000 or 16000 ONLY after you hear it at 500.
const int sampleRate = 8000;  // sample rate in Hz

i2s_data_bit_width_t bps = I2S_DATA_BIT_WIDTH_16BIT;
i2s_mode_t mode = I2S_MODE_STD;
i2s_slot_mode_t slot = I2S_SLOT_MODE_STEREO;

const unsigned int halfWavelength = sampleRate / frequency / 2;  // half wavelength of square wave

int16_t sample = amplitude;  // current sample value
unsigned int count = 0;

I2SClass i2s;

void setup() {
  Serial.begin(115200);
  Serial.println("I2S simple tone");

  i2s.setPins(I2S_BCLK, I2S_LRC, I2S_DIN);

  // start I2S at the sample rate with 16-bits per sample
  if (!i2s.begin(mode, sampleRate, bps, slot)) {
    Serial.println("Failed to initialize I2S!");
    while (1);  // do nothing
  }
}

void loop() {
  if (count % halfWavelength == 0) {
    // invert the sample every half wavelength count multiple to generate square wave
    sample = -1 * sample;
  }

  int16_t left = sample;
  int16_t right = sample;
  i2s.write(&left, sizeof(left));
  i2s.write(&right, sizeof(right));

  // increment the counter for the next sample
  count++;
}
