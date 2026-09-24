# REFERENCES SLIDE — text and links kept separate

## PART A — SLIDE TEXT  (hyperlink each bold title to its link in Part B)

**The model we build on — and measure against**
- **GTCRN, ICASSP 2024** — 23.7k params. The architecture RHEAR derives from, and
  the baseline we beat on defence noise using its authors' own weights.
- **SepFormer, 2021** — the 25.6M-parameter transformer we beat by 5.2 dB SI-SDR
  with 49,663 parameters.
- **MetricGAN+, 2021** — trained on PESQ. Best PESQ on our set, negative SI-SDR.
  Our evidence that one metric decides nothing.

**The physics that forces the architecture**
- **Kuo & Morgan, Proc. IEEE 1999** — the FxLMS derivation L0 implements and the
  causality condition the design obeys.
- **Causality study, ANC headset** — source of our 146 µs budget and 7 cm
  microphone geometry.
- **ADAU1772 datasheet** — the 38 µs group delay that selected the codec.

**Classical estimators running on our chip**
- **Ephraim & Malah, 1984** — decision-directed log-MMSE. Took our on-device
  suppression from 7.5 dB to 14.9 dB.
- **Martin, 2001** — minimum-statistics noise estimation; its sliding window
  stopped our noise floor collapsing to zero.
- **Sohn et al., 1999** — statistical VAD. Measured at d′ 0.97 against energy's
  5.11, and we report that it lost.

**Benchmark and the human case**
- **VoiceBank+DEMAND** — the benchmark the PS thresholds come from; we meet all
  three on its 824 clips.
- **NIHL in Indian Air Force personnel** — 22.9% overall, 26.18% technical
  trades. Why this project exists.

---

## PART B — LINKS  (for hyperlinking the titles above)

1. GTCRN, ICASSP 2024
   https://ieeexplore.ieee.org/document/10448310
   code: https://github.com/Xiaobin-Rong/gtcrn

2. SepFormer — "Attention Is All You Need in Speech Separation"
   https://arxiv.org/abs/2010.13154

3. MetricGAN+
   https://arxiv.org/abs/2104.03538

4. Kuo & Morgan — Active Noise Control: A Tutorial Review, Proc. IEEE 1999
   http://www2.coe.pku.edu.cn/tpic/2010913102917710.pdf

5. Causality study on a feedforward ANC headset, free field
   https://opus.lib.uts.edu.au/bitstream/10453/118176/4/Causality%20study%20on%20a%20feedforward%20active%20noise%20control%20headset%20with%20different%20noise%20coming%20directions%20in%20free%20field.pdf

6. Analog Devices ADAU1772 datasheet
   https://www.analog.com/media/en/technical-documentation/data-sheets/ADAU1772.pdf

7. Ephraim & Malah — Speech enhancement using a minimum mean-square error
   log-spectral amplitude estimator, IEEE Trans. ASSP 33(2):443-445, 1985
   (and ASSP 32(6):1109-1121, 1984)
   https://ieeexplore.ieee.org/document/1164453

8. Martin — Noise power spectral density estimation based on optimal smoothing
   and minimum statistics, IEEE Trans. Speech & Audio 9(5):504-512, 2001
   https://ieeexplore.ieee.org/document/928915

9. Sohn, Kim & Sung — A statistical model-based voice activity detection,
   IEEE Signal Processing Letters 6(1):1-3, 1999
   https://ieeexplore.ieee.org/document/736233

10. VoiceBank+DEMAND (Valentini-Botinhao et al.), Edinburgh DataShare
    https://datashare.ed.ac.uk/handle/10283/2791

11. Prevalence of Noise Induced Hearing Loss in Indian Air Force Personnel
    https://pubmed.ncbi.nlm.nih.gov/27408258/

### Supporting, if a second slide is available
12. Espressif esp-sr — the neural VAD running on our ESP32-S3
    https://github.com/espressif/esp-sr
13. Espressif esp-dsp — assembly-optimised FFT/FIR for the S3
    https://github.com/espressif/esp-dsp
14. LiSenNet — current lightweight-SE efficiency frontier
    https://arxiv.org/pdf/2409.13285
15. IMSE, 2025 — 0.427M-parameter SOTA; 427 KB at int8, outside the PS's 200 KB cap
    https://arxiv.org/abs/2511.14515
16. iDEX — Innovations for Defence Excellence
    https://idex.gov.in/
17. Defence production FY2025-26 — PIB
    https://www.pib.gov.in/PressReleasePage.aspx?PRID=2273824

### A note on entries 7, 8 and 9
These are the three classical estimators actually implemented in our firmware,
and each one is cited because a measurement in this project turned on it, not
because it is famous. Full journal citations are given so they resolve even if
a publisher URL changes.
