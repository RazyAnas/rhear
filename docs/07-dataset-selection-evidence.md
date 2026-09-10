# Dataset selection, from the literature

Written 2026-09-06, after G8 was accepted and G9/G10 were rejected.

## Why this document exists

Three architecture-side levers were tested and eliminated by measurement:

| lever | experiment | result |
|---|---|---|
| frequency resolution | G8, 48 -> 96 bands | helped; KEPT |
| model capacity | G9, +56% params | TRAIN PESQ 2.049 -> 2.050. no effect |
| the objective | G10, rho 8 -> 4 | TRAIN PESQ -> 2.068; STOI/SI-SAR regressed. REJECTED |

The train-val gap never moved: TRAIN PESQ 2.068 / VAL 1.769 / TEST 1.678.
The model fits its own training data far better than held-out data, and no
change to the network altered that. The remaining lever is the data.

## The measured cause

    h3_20k train split: 20,000 clips drawn from 26 SPEAKERS.

`registry.py` built h3_20k from SMALL_BUILD, which omits
`librispeech_train_clean_100`. The train pool was LibriSpeech dev-clean +
test-clean -- 80 speakers total, 26 of which landed in the train split.
DEFAULT_BUILD already lists train-clean-100; it was simply never fetched.

## What the literature says to scale

"Scale This, Not That: Investigating Key Dataset Attributes for Efficient
Speech Enhancement Scaling" (arXiv 2412.14890):

  * speaker diversity improves performance but SATURATES BEYOND ~100 SPEAKERS,
    limited by total data size and model capacity. Our model is 49,663 params,
    squarely in that regime.
  * noise VARIETY is what keeps paying: "merely expanding noise duration
    without variety showed limited impact", while expanding the number of
    noise TYPES improves performance "particularly on unseen noise".
  * text and language diversity have negligible effect; deprioritise.

CONSEQUENCE, and it reversed an earlier plan of ours: train-clean-100's 251
speakers already clear the saturation point. train-clean-360 (+921 speakers)
and train-other-500 (+1,166) would cost 53 GB for close to nothing. The budget
belongs in NOISE CLASSES.

## Why impulsive noise gets extra weight

Our worst SNR cut is < -5 dB (E_def PESQ +0.14; realnoise +0.02), and our noise
classes are gunshot, shelling, fighter, helicopter -- transient, not stationary.
The impulsive-noise literature (arXiv 1910.02710) records that noise PSD is
systematically underestimated for highly non-stationary noise, and that impulse
sound databases are scarce. So the noise pool is deliberately weighted toward
impulsive event classes rather than more stationary hum.

## Selection, ordered by value per GB

 1. NOISEX-92            0.1 GB   F16, Buccaneer, Leopard/M109 tank, machine
                                  gun, destroyer engine + ops room. Our PS's
                                  own noise classes, and a standard benchmark.
 2. MAD                  1.1 GB   gunshot/shelling/helicopter/fighter/vehicle/
                                  footsteps, real military footage.
                                  'communication' class EXCLUDED (contains
                                  speech; would contaminate the target).
 3. RIRS_NOISES          1.2 GB   real measured rooms.
 4. LibriSpeech tc-100   6.3 GB   251 speakers -- clears the ~100 saturation.
 5. dev-clean+test-clean 0.7 GB   speaker-disjoint val/test pools.
 6. MUSAN noise         10.3 GB   registry dependency, 6 broad classes.
 7. FSD50K             ~32 GB     200 AudioSet-ontology classes, 51,197 clips.
 8. DNS5 noise_fullband ~40 GB    150 noise classes (AudioSet + Freesound).
 9. WHAM! noise         13 GB     real bar/cafe ambience, native 16 kHz.
10. ESC-50 + UrbanSound8K 7 GB    +60 classes incl. gunshot, siren.

Items 1-6 are ~22 GB and already deliver 251 speakers, military-specific noise
and real RIRs. The ordering is deliberate: stopping early still leaves the
highest-value set on disk.

## Licence consequence

Items 9 and 10, and DEMAND inside item 8, are CC BY-NC / CC BY-SA. Including
them forces the derived dataset to CC BY-NC, which `registry.derived_licence()`
will report and which blocks a production path. Mitigation: build TWO manifests
from ONE download pool -- a permissive-only manifest for the record, and a full
manifest for the best numbers. The audio is shared, so this costs nothing.

## What does NOT change

The architecture is frozen. Every corpus is resampled to 16 kHz mono and cut to
the same 4 s manifest contract h3_20k already uses. The model stays 49,663
params, 96 bands, hop 256 -- inside the ESP32-S3 N16R8's 16 MB flash and 8 MB
PSRAM. Weight storage at int8 is ~48.5 KB; 200 KB is the PS *cap*, not our
footprint, and an earlier draft of this line conflated the two.

**Correction.** This paragraph previously ended "so the psram_test.c port stays
valid." That is wrong and was wrong when written. `psram_test.c:12` builds at
`N_BANDS 48` -- it ports G7. Every model from G8 onward is 96 bands, so the
existing C port does NOT cover the shipping model, and re-porting is real work
rather than a recompile. The port is also front-half only (encoder, full-band
branch, fuse, spp); the DPRNN, decoder and ISTFT are not in it. See
`demo/README.md:76-79`, which has always stated this correctly.

## Sources

  * arXiv 2412.14890 -- Scale This, Not That
  * arXiv 1910.02710 -- Impulsive Noise Detection for Intelligibility
  * Nature Scientific Data 2024 -- A Military Audio Dataset (MAD)
  * NOISEX-92, Speech Communication 12(3)
  * microsoft/DNS-Challenge -- noise_fullband, 58 GB
  * FSD50K, arXiv 2010.00475 -- 51,197 clips, 200 classes

## EDA before training G12 (2026-09-07)

Running the data through a distribution check BEFORE training caught an error
that would have wasted the run and probably read as "noise variety does not
help".

### What the eval sets actually contain

    E_def      gunshot 15.4%  footsteps 13.5%  drone 12.5%  fighter 12.5%
               helicopter 12.2%  environmental 12.1%  vehicle 11.4%
               shelling 10.3%          -> 88% MILITARY
    realnoise  environmental 100%

### What each training set contains

                        distinct classes   military share
    G11 (h3-style)               9              87.9%      <- matches E_def
    G12 (uniform sampling)     231              13.7%      <- 74 points off

The G12 noise pool is 46,872 clips of which 37,582 (80%) are FSD50K. build()
samples the pool UNIFORMLY, so raw pool proportions become training
proportions: a defence noise canceller trained on 80% generic Freesound audio
and then graded on gunfire and rotor noise.

Everything else was already well matched, which is why only the composition
needed fixing:

                    train    E_def
    SNR dB           4.96     4.63
    speech frac      0.598    0.582
    clipping        15.6%    13.2%
    reverb          59.8%    62.3%
    layers/clip      2.00     2.11

### Fix: weighted sampling, not a smaller pool

build() picks uniformly from the list it is handed, so the LIST carries the
weighting. Entries are replicated to hit target shares:

    mad 0.65   noisex 0.10   musan 0.05
    fsd50k 0.13   esc50 0.04   dns5 0.02   us8k 0.01

Military stays ~75% (E_def is 88%, G11 trained at 86%), while the non-military
slice -- which in G11 was MUSAN's ~4 classes -- becomes ~230 classes. Diversity
is bought INSIDE the stratum rather than by replacing the stratum.

### Speech contamination, three corpora

Speech in the noise pool teaches the model to suppress the signal it must
preserve. Excluded explicitly:

    MAD label 0 'communication'      981 clips  (verified by class order AND by
                                     measured 2-8 Hz syllabic modulation:
                                     0.413 vs 0.19-0.33 for every other label)
    FSD50K 15 speech/vocal classes  3384 clips
    ESC-50 human-vocal categories    240 clips
    UrbanSound8K children_playing      -- excluded (shouting voices)

### Raw-corpus EDA (2026-09-07)

    corpus        n      rate   dur med   <4s   <1s  silent
    MUSAN        930    16000    10.7s    20%    5%   0.0%
    MAD         7466    16000     5.0s    31%    0%   0.0%
    NOISEX       115     8000     3.4s    60%    2%   0.0%
    ESC-50      2000    44100     5.0s     0%    0%   0.4%
    FSD50K     40966    44100     4.7s    44%   12%   2.0%

NOISEX-92 is 8 kHz: the f16/Buccaneer/Leopard/M109/machinegun benchmark carries
NO energy above 4 kHz. Upsampling to 16 kHz does not create any. At ~10% of the
weighted pool this is acceptable, but those classes are band-limited and should
not be cited as full-band evidence.

### BUG: impulsive noise was being tiled

mixing.IMPULSIVE_CLASSES matches the class name EXACTLY:

    {"gunshot", "shelling", "footsteps", "impulse", "blast"}

The Kaggle MAD layout has no class directories, so build_g11/g12 named classes
mad_1..mad_6 from the CSV label column. NONE of them matched, so every sub-4s
gunshot/shelling/footsteps clip was TILED -- turning one blast into a periodic
impulse train, the precise artefact mixing._fit was written to prevent. 31% of
MAD clips are shorter than the 4 s window, and MAD was 88% of G11's noise.

E_def was built from the ORIGINAL directory-named MAD layout, so its impulsive
handling is correct. Training therefore contained an artefact the eval set does
not -- a train/test mismatch, not merely unrealistic audio.

Fixed by mapping to canonical names (verified acoustically, see above):
    1->gunshot  2->footsteps  3->shelling  4->vehicle  5->helicopter  6->fighter
and normalising ESC-50 gun_shot, UrbanSound8K gun_shot and FSD50K
Gunshot_and_gunfire / Explosion to the same canonical names.

G11's +0.016 PESQ over G8 was obtained DESPITE this bug.

### Residual speech after label filtering

Label filtering removed 3,384 FSD50K clips. Measuring 2-8 Hz syllabic
modulation on samples of both sides:

    KEPT      mean 0.342   12% above the 0.42 speech threshold
    EXCLUDED  mean 0.381   33% above

So the labels do separate speech, but ~12% of kept clips remain speech-like --
about 4,400 of 37,582. At FSD50K's 13% weight that is ~1.6% of all noise layers.
The metric over-triggers on music and applause, so 12% is an upper bound.
Accepted as a known limitation rather than screening 37,582 clips by hand.
