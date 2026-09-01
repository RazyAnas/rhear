#!/usr/bin/env python3
"""Assert an evaluation set is disjoint from every training set.

Run this after building any dataset, and before trusting any metric.

    python3 check_leakage.py --eval <eval_dir> --train <dir> [<dir> ...]

Why this exists. partG_build_real.py argued no audit was needed because
the model then trained on 100% synthetic noise, so no real recording
could overlap. That stopped being true when we started training on real
MUSAN noise, and nothing enforced it afterwards.

One trap worth naming, because it cost an afternoon: build() emits
throwaway 1-clip 'train' and 'val' rows beside the real 'test' split,
because n_per_split requires all three. Those dummy clips draw from the
whole pool and WILL collide with training sets. They are never
evaluated. Compare the TEST split only -- comparing all rows reports
leakage that does not exist.
"""
import os, sys, json, argparse

EVAL_SPLIT = "test"
TRAIN_SPLITS = ("train", "val")


def read(path):
    with open(path) as f:
        return [json.loads(l) for l in f][1:]


def sources(rows, splits):
    """Noise-source ids and RIR ids used by the given splits."""
    rs = [r for r in rows if r["split"] in splits]
    src = {l["source"] for r in rs for l in r.get("noise_layers", [])}
    rir = {r["rir_id"] for r in rs if r.get("rir_id")}
    return src, rir


def manifest_of(d):
    p = d if d.endswith(".jsonl") else os.path.join(d, "manifest.jsonl")
    if not os.path.exists(p):
        sys.exit("no manifest at %s" % p)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    help="dataset dir (or manifest) holding the eval split")
    ap.add_argument("--train", required=True, nargs="+",
                    help="one or more training dataset dirs")
    a = ap.parse_args()

    ev = read(manifest_of(a.eval))
    e_src, e_rir = sources(ev, (EVAL_SPLIT,))
    n_eval = len([r for r in ev if r["split"] == EVAL_SPLIT])
    print("  eval  %s" % a.eval)
    print("    %d clips in '%s' split, %d noise sources, %d RIRs"
          % (n_eval, EVAL_SPLIT, len(e_src), len(e_rir)))
    if not n_eval:
        sys.exit("  eval split '%s' is empty" % EVAL_SPLIT)

    bad = []
    for d in a.train:
        tr = read(manifest_of(d))
        t_src, t_rir = sources(tr, TRAIN_SPLITS)
        ns, nr = e_src & t_src, e_rir & t_rir
        ok = not ns and not nr
        print("    vs %-40s noise %d, rir %d   %s"
              % (os.path.basename(d.rstrip("/")), len(ns), len(nr),
                 "OK" if ok else "*** LEAKAGE ***"))
        if not ok:
            bad.append((d, sorted(ns)[:5], sorted(nr)[:5]))

    if bad:
        print()
        for d, ns, nr in bad:
            print("  LEAKAGE vs %s" % d)
            if ns:
                print("    noise sources: %s" % ns)
            if nr:
                print("    rir ids:       %s" % nr)
        sys.exit(1)
    print("\n  CLEAN - eval split is disjoint from every training set given")
