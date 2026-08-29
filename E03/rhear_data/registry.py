"""Dataset registry with VERIFIED licence terms.

Every entry was checked against the dataset's own page or licence file on
2026-08-28. `verified` records what was actually confirmed; anything unconfirmed
says so rather than guessing.

Licence policy for RHEAR
------------------------
This is a DRDO / iDEX project with a production path, so the DEFAULT build uses
only attribution-style licences: CC BY 4.0, CC0, US Public Domain, Apache 2.0.

Two well-known corpora are DELIBERATELY EXCLUDED by default:

  * ESC-50 and UrbanSound8K are CC BY-NC 3.0 -- non-commercial. An NC term
    contaminates any downstream product use.
  * DEMAND is CC BY-SA 3.0 -- share-alike is viral: a dataset derived from it
    must itself be CC BY-SA, which is not acceptable for defence work.

Both can be enabled explicitly (`include_restricted=True`) for research-only
experiments, and the manifest then records the resulting obligation.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Dataset:
    key: str
    name: str
    role: str                      # clean_speech | noise | rir
    url: str
    licence: str
    licence_ok: bool               # permissive enough for the default build
    size_gb: Optional[float]
    content_kind: str              # "real recording" | "synthetic"
    verified: str                  # what was actually confirmed, and when
    notes: str = ""
    classes: List[str] = field(default_factory=list)


DATASETS = {
    # ---------------- clean speech ----------------
    "librispeech_dev_clean": Dataset(
        key="librispeech_dev_clean", name="LibriSpeech dev-clean", role="clean_speech",
        url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
        licence="CC BY 4.0", licence_ok=True, size_gb=0.31,
        content_kind="real recording",
        verified="licence CC BY 4.0 (openslr.org/12); size 0.31 GB measured by HTTP "
                 "content-length 2026-08-28",
        notes="40 speakers. Used as the VALIDATION speaker pool."),
    "librispeech_test_clean": Dataset(
        key="librispeech_test_clean", name="LibriSpeech test-clean", role="clean_speech",
        url="https://www.openslr.org/resources/12/test-clean.tar.gz",
        licence="CC BY 4.0", licence_ok=True, size_gb=0.35,
        content_kind="real recording",
        verified="licence CC BY 4.0 (openslr.org/12)",
        notes="40 speakers. Reserved for the HELD-OUT REAL-WORLD TEST SET."),
    "librispeech_train_clean_100": Dataset(
        key="librispeech_train_clean_100", name="LibriSpeech train-clean-100",
        role="clean_speech",
        url="https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        licence="CC BY 4.0", licence_ok=True, size_gb=6.3,
        content_kind="real recording",
        verified="licence CC BY 4.0; 100 h, 251 speakers (openslr.org/12)",
        notes="251 speakers, 100 h. TRAIN speaker pool. LibriSpeech's own splits are "
              "already speaker-disjoint, which makes leakage prevention exact."),

    # ---------------- real environmental noise ----------------
    "musan_noise": Dataset(
        key="musan_noise", name="MUSAN (noise subset)", role="noise",
        url="https://www.openslr.org/resources/17/musan.tar.gz",
        licence="per-file CC or US Public Domain", licence_ok=True, size_gb=10.32,
        content_kind="real recording",
        verified="~109 h total, 930 noise files, 16 kHz WAV;per-file LICENSE files "
                 "(openslr.org/17). Size 10.32 GB measured by HTTP content-length "
                 "2026-08-28",
        notes="Only the noise/ subtree is used (free-sound + sound-bible). Real "
              "recorded environmental and machinery noise. The tarball is not "
              "separable server-side, so the whole 10.3 GB is fetched once and "
              "cached to Drive.",
        classes=["ambient", "machinery", "technical", "crowd"]),

    # ---------------- military noise ----------------
    "mad": Dataset(
        key="mad", name="MAD - Military Audio Dataset", role="noise",
        url="https://github.com/kaen2891/military_audio_dataset",
        licence="CC BY 4.0", licence_ok=True, size_gb=1.4,
        content_kind="real recording",
        verified="CC BY 4.0 per the AUTHORS' REPOSITORY, which states explicitly: "
                 "'This project, including the dataset, code, and paper, is licensed "
                 "under CC BY 4.0' (github.com/kaen2891/military_audio_dataset). "
                 "8,075 clips, ~12 h, 16 kHz mono WAV, 1-10 s, 5 annotators, real "
                 "military footage with games/fiction excluded (Scientific Data 2024). "
                 "DISCREPANCY: the Kaggle mirror's licence dropdown says CC-BY-SA-4.0. "
                 "The repository statement is more specific and explicitly covers the "
                 "DATASET, so it governs -- but if MAD ever enters a commercial "
                 "release, confirm with the authors in writing. NOTE: an earlier "
                 "version of this entry cited the PAPER's CC BY 4.0, which covers the "
                 "ARTICLE only, not the data. Right answer, wrong evidence.",
        notes="USED AS A MILITARY-NOISE SOURCE ONLY. MAD is a CLASSIFICATION corpus "
              "with no clean-speech pairing, so it is never treated as a "
              "speech-enhancement pair. Its 'communication' class contains speech "
              "and is EXCLUDED from the noise pool to avoid contaminating the "
              "enhancement target.",
        classes=["gunshot", "footsteps", "shelling", "vehicle", "helicopter",
                 "fighter", "communication(excluded)"]),

    # ---------------- room acoustics ----------------
    "rirs_noises": Dataset(
        key="rirs_noises", name="RIRS_NOISES (OpenSLR 28)", role="rir",
        url="https://www.openslr.org/resources/28/rirs_noises.zip",
        licence="Apache 2.0", licence_ok=True, size_gb=1.22,
        content_kind="real recording (real RIR subset) + synthetic (simulated subset)",
        verified="Apache 2.0 (openslr.org/28); contains REAL measured RIRs from RWCP, "
                 "REVERB-2014 and AIR, plus SIMULATED RIRs and MUSAN point-source "
                 "noises. Size 1.22 GB measured by HTTP content-length 2026-08-28",
        notes="Only the REAL measured RIRs (real_rirs_isotropic_noises) are used by "
              "default, so reverberation in the dataset comes from real rooms. The "
              "simulated subset is available but tagged differently in the manifest."),

    # ---------------- optional / restricted ----------------
    "dns_noise": Dataset(
        key="dns_noise", name="DNS Challenge noise (optional)", role="noise",
        url="https://github.com/microsoft/DNS-Challenge",
        licence="mixed: AudioSet CC BY 4.0 + Freesound CC0 + DEMAND CC BY-SA 3.0",
        licence_ok=False, size_gb=None,
        content_kind="real recording",
        verified="noise drawn from AudioSet (CC BY 4.0), Freesound (CC0 only) and "
                 "DEMAND (CC BY-SA 3.0); full repo is ~827 GB",
        notes="NOT in the default build. The DEMAND component is CC BY-SA, which is "
              "viral, and the full repository is far too large. MUSAN covers the same "
              "material with cleaner terms. Enable only with a size cap and only the "
              "AudioSet/Freesound tarballs if DNS noise is specifically wanted."),
    "esc50": Dataset(
        key="esc50", name="ESC-50", role="noise",
        url="https://github.com/karolpiczak/ESC-50",
        licence="CC BY-NC 3.0", licence_ok=False, size_gb=0.6,
        content_kind="real recording",
        verified="CC BY-NC 3.0 (repo LICENSE) -- NON-COMMERCIAL",
        notes="EXCLUDED by default: the NC term is incompatible with a production "
              "path."),
    "urbansound8k": Dataset(
        key="urbansound8k", name="UrbanSound8K", role="noise",
        url="https://zenodo.org/records/1203745",
        licence="CC BY-NC 3.0", licence_ok=False, size_gb=6.6,
        content_kind="real recording",
        verified="CC BY-NC 3.0 -- NON-COMMERCIAL",
        notes="EXCLUDED by default, same reason as ESC-50."),
    "demand": Dataset(
        key="demand", name="DEMAND", role="noise",
        url="https://zenodo.org/records/1227121",
        licence="CC BY-SA 3.0", licence_ok=False, size_gb=None,
        content_kind="real recording",
        verified="CC BY-SA 3.0 -- SHARE-ALIKE",
        notes="EXCLUDED by default: share-alike would force the derived RHEAR "
              "dataset to be CC BY-SA."),
}

DEFAULT_BUILD = ["librispeech_train_clean_100", "librispeech_dev_clean",
                 "librispeech_test_clean", "musan_noise", "mad", "rirs_noises"]

SMALL_BUILD = ["librispeech_dev_clean", "librispeech_test_clean",
               "musan_noise", "mad", "rirs_noises"]


def licence_table(keys=None):
    ks = keys or list(DATASETS)
    rows = [("dataset", "role", "licence", "ok", "GB", "content")]
    for k in ks:
        d = DATASETS[k]
        rows.append((d.name, d.role, d.licence,
                     "yes" if d.licence_ok else "NO",
                     f"{d.size_gb:.2f}" if d.size_gb else "?",
                     d.content_kind))
    return rows


def derived_licence(keys):
    """What licence must the RHEAR derived dataset carry, given its inputs?"""
    lic = {DATASETS[k].licence for k in keys}
    if any("BY-SA" in x for x in lic):
        return "CC BY-SA 4.0 (forced by a share-alike input)"
    if any("NC" in x for x in lic):
        return "CC BY-NC 4.0 (forced by a non-commercial input) - NOT usable in a product"
    return "CC BY 4.0 with attribution to each source corpus"
