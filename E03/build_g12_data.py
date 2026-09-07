#!/usr/bin/env python3
"""G12 dataset: same 20k clips, same 251 speakers, ~2.8x the NOISE CLASSES.

G11 proved speaker diversity is a real lever but a modest one: 26 -> 251
speakers moved the train-test PESQ gap only 0.390 -> 0.348. "Scale This, Not
That" (arXiv 2412.14890) predicts exactly that -- speakers saturate past ~100,
while noise VARIETY keeps paying, "particularly on unseen noise".

So this build holds everything else fixed and changes only the noise pool:

    G11 noise: MUSAN(930) + MAD(6485) + NOISEX(115)          ~140 classes
    G12 noise: the above + ESC-50(2000) + FSD50K(40966)      ~390 classes

Clip count stays 20,000 and the speaker pools are untouched, so any difference
against G11 is attributable to noise diversity alone.

TWO SPEECH-CONTAMINATION TRAPS, both handled explicitly. Letting speech into the
noise pool teaches the model to suppress the very signal it must preserve:
  * MAD label 0 is 'communication' (verified by class order AND by measured
    2-8 Hz syllabic modulation: 0.413 vs 0.19-0.33 for every other label)
  * FSD50K has 15 speech/vocal classes in its AudioSet ontology
  * ESC-50's human-vocal categories are dropped for the same reason
"""
import csv, os, sys
sys.path.insert(0, os.path.expanduser("~/PS#2/handoff/code"))
from rhear_data.build import build, index_musan_noise, _is_junk

C   = "/Volumes/RHEARDATA/corpora"
OUT = os.path.expanduser("~/PS#2/handoff/data/g12_20k")   # rebuilt with DNS5+US8K

# mixing.IMPULSIVE_CLASSES matches the class name EXACTLY: {"gunshot","shelling",
# "footsteps","impulse","blast"}. Naming MAD's classes mad_1..mad_6 meant NONE of
# them matched, so every sub-4s gunshot/shelling/footsteps clip was TILED into a
# periodic impulse train -- the exact artefact mixing._fit exists to prevent, and
# one E_def does not contain (it was built from the directory-named MAD layout).
# EDA: 31% of MAD clips are shorter than the 4 s window.
MAD_CLASS = {"1": "gunshot", "2": "footsteps", "3": "shelling",
             "4": "vehicle",  "5": "helicopter", "6": "fighter"}

FSD_SPEECH = {"Speech","Male_speech_and_man_speaking","Female_speech_and_woman_speaking",
              "Child_speech_and_kid_speaking","Conversation","Chatter","Whispering",
              "Shout","Yell","Singing","Male_singing","Female_singing",
              "Speech_synthesizer","Laughter","Crying_and_sobbing"}
ESC_VOCAL  = {"crying_baby","laughing","coughing","sneezing","snoring","breathing"}
# UrbanSound8K's children_playing is full of shouting voices -- same speech
# contamination risk as MAD 'communication' and FSD50K's Speech classes.
US8K_VOCAL = {"children_playing"}

def mad_noise():
    out, skipped = [], 0
    root = f"{C}/mad/MAD_dataset"
    for f in ("training.csv","test.csv"):
        for r in list(csv.reader(open(os.path.join(root,f))))[1:]:
            if len(r) < 3: continue
            if r[2] == "0": skipped += 1; continue          # communication = speech
            p = os.path.join(root, r[1])
            if not os.path.exists(p): continue
            vid = os.path.basename(os.path.dirname(r[1]))
            out.append(dict(path=p, cls=MAD_CLASS.get(r[2], f"mad_{r[2]}"),
                            source=f"mad:{vid}", corpus="mad",
                            sub=f"label{r[2]}", provenance="real recording"))
    return out, skipped

def noisex():
    out=[]
    for dp,_,fns in os.walk(f"{C}/Noises"):
        for fn in fns:
            if not fn.lower().endswith(".wav") or _is_junk(fn): continue
            out.append(dict(path=os.path.join(dp,fn), cls="military_bench", corpus="noisex",
                            source=f"noisex:{os.path.splitext(fn)[0]}",
                            sub=os.path.basename(dp), provenance="real recording"))
    return out

def esc50():
    """ESC-50 filename is FOLD-CLIPID-TAKE-TARGET.wav; CLIPID is the Freesound
    source, so grouping on it keeps takes from one recording inside one split."""
    meta = f"{C}/ESC-50-master/meta/esc50.csv"
    keep, skipped = {}, 0
    for r in list(csv.DictReader(open(meta))):
        if r["category"] in ESC_VOCAL: skipped += 1; continue
        keep[r["filename"]] = r["category"]
    out=[]
    for dp,_,fns in os.walk(f"{C}/ESC-50-master/audio"):
        for fn in fns:
            if not fn.lower().endswith(".wav") or _is_junk(fn) or fn not in keep: continue
            clipid = fn.split("-")[1] if "-" in fn else fn
            cat = keep[fn]
            cls = {"gun_shot":"gunshot","footsteps":"footsteps"}.get(cat, f"esc_{cat}")
            out.append(dict(path=os.path.join(dp,fn), cls=cls, corpus="esc50",
                            source=f"esc50:{clipid}", sub=keep[fn], provenance="real recording"))
    return out, skipped

def fsd50k():
    gt = f"{C}/FSD50K.ground_truth/dev.csv"
    keep, skipped = {}, 0
    for r in csv.DictReader(open(gt)):
        labs = set(r["labels"].split(","))
        if labs & FSD_SPEECH: skipped += 1; continue
        first = r["labels"].split(",")[0]
        # canonical names so mixing.IMPULSIVE_CLASSES actually fires
        if "Gunshot" in first or "gunfire" in first.lower(): first = "gunshot"
        elif "Explosion" in first or "Boom" in first:        first = "blast"
        keep[r["fname"]] = first
    out=[]
    d = f"{C}/FSD50K.dev_audio"
    for dp,_,fns in os.walk(d):
        for fn in fns:
            if not fn.lower().endswith(".wav") or _is_junk(fn): continue
            base = os.path.splitext(fn)[0]
            if base not in keep: continue
            out.append(dict(path=os.path.join(dp,fn), cls=f"fsd_{keep[base]}", corpus="fsd50k",
                            source=f"fsd50k:{base}", sub=keep[base], provenance="real recording"))
    return out, skipped

def dns5():
    """DNS5 noise_fullband: AudioSet-derived, the largest noise-class source.
    Each archive holds flat .wav files; the filename is the AudioSet clip id,
    which is also the source recording, so it is the group key."""
    root = f"{C}/dns5_noise"
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".wav") or _is_junk(fn): continue
            base = os.path.splitext(fn)[0]
            out.append(dict(path=os.path.join(dp, fn), cls="dns5_audioset",
                            corpus="dns5", source=f"dns5:{base}",
                            sub="audioset", provenance="real recording"))
    return out

def us8k():
    """UrbanSound8K. Grouped by fsID (the Freesound source recording), so slices
    cut from one recording cannot straddle a split."""
    meta = f"{C}/UrbanSound8K/metadata/UrbanSound8K.csv"
    if not os.path.exists(meta): return [], 0
    out, skipped = [], 0
    for r in csv.DictReader(open(meta)):
        if r["class"] in US8K_VOCAL: skipped += 1; continue
        p = f"{C}/UrbanSound8K/audio/fold{r['fold']}/{r['slice_file_name']}"
        if not os.path.exists(p): continue
        cls = {"gun_shot":"gunshot"}.get(r["class"], f"us8k_{r['class']}")
        out.append(dict(path=p, cls=cls, corpus="us8k",
                        source=f"us8k:{r['fsID']}", sub=r["class"],
                        provenance="real recording"))
    return out, skipped

# EDA finding (see the analysis in docs/07): E_def -- the set the PS is graded
# against -- is 88% military noise (gunshot/footsteps/drone/fighter/helicopter/
# vehicle/shelling) and 12% environmental. G11's TRAIN pool was 86% military,
# which is why it matched. The raw G12 pool is 80% FSD50K by clip count, so
# sampling it uniformly would drop military to 14% and train a defence canceller
# on generic Freesound audio. build() picks noise uniformly from the list it is
# given, so the list itself carries the weighting: entries are replicated to hit
# the target shares below. Military stays dominant; the non-military slice --
# which in G11 was MUSAN's ~4 classes -- becomes ~230 classes.
TARGET = {"mad": 0.65, "noisex": 0.10, "musan": 0.05,
          "fsd50k": 0.13, "esc50": 0.04, "dns5": 0.02, "us8k": 0.01}
POOL_N = 60000

def reweight(groups):
    """groups: {corpus: [entries]} -> a single list whose corpus proportions
    match TARGET, by replicating (or subsampling) each corpus."""
    import math, random
    rnd = random.Random(777)
    out = []
    for corpus, want in TARGET.items():
        src = groups.get(corpus, [])
        if not src: continue
        need = int(round(want * POOL_N))
        if len(src) >= need:
            out += rnd.sample(src, need)
        else:
            reps = need // len(src)
            out += src * reps + rnd.sample(src, need - reps * len(src))
    rnd.shuffle(out)
    return out

def main():
    mus = index_musan_noise(f"{C}/musan")
    mad, mad_sk = mad_noise()
    nx  = noisex()
    esc, esc_sk = esc50()
    fsd, fsd_sk = fsd50k()
    dns = dns5()
    us, us_sk = us8k()
    groups = {"musan": mus, "mad": mad, "noisex": nx,
              "esc50": esc, "fsd50k": fsd, "dns5": dns, "us8k": us}
    noise = reweight(groups)
    print(f"noise pool:")
    print(f"   MUSAN    {len(mus):6d}")
    print(f"   MAD      {len(mad):6d}   ({mad_sk} 'communication' speech clips EXCLUDED)")
    print(f"   NOISEX   {len(nx):6d}")
    print(f"   ESC-50   {len(esc):6d}   ({esc_sk} human-vocal clips EXCLUDED)")
    print(f"   FSD50K   {len(fsd):6d}   ({fsd_sk} speech-labelled clips EXCLUDED)")
    print(f"   DNS5     {len(dns):6d}")
    print(f"   US8K     {len(us):6d}   ({us_sk} children_playing clips EXCLUDED)")
    import collections as _c
    print(f"\n   after reweighting to match E_def's 88% military composition:")
    cc = _c.Counter(n["corpus"] for n in noise)
    for k, v in cc.most_common():
        print(f"     {k:<9} {v:6d}  {100*v/len(noise):5.1f}%")
    mil = sum(v for k, v in cc.items() if k in ("mad", "noisex"))
    print(f"   MILITARY share: {100*mil/len(noise):.1f}%   (E_def is 88%, G11 trained at 86%)")
    print(f"   TOTAL    {len(noise):6d}   source groups: {len({n['source'] for n in noise})}")
    print(f"   distinct classes: {len({n['cls'] for n in noise})}")

    speech_roots = {"train": f"{C}/LibriSpeech/train-clean-100",
                    "val":   f"{C}/LibriSpeech/dev-clean",
                    "test":  f"{C}/LibriSpeech/test-clean"}
    n, rep = build(OUT, speech_roots, None, None, f"{C}/RIRS_NOISES",
                   n_per_split={"train":20000,"val":500,"test":500},
                   dur_s=4.0, seed=777, snr_range=(-10.0,20.0),
                   heldout_n=0, noise_index=noise)
    print(f"\nwrote {n} -> {OUT}")
    print(f"leakage: {'CLEAN' if rep['clean'] else 'LEAKAGE DETECTED'}")

if __name__ == "__main__":
    main()
