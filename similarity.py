"""
similarity.py — Speaker Similarity & Clone Quality Metrics
═══════════════════════════════════════════════════════════
Two metrics:
  1. GE2E Speaker Similarity (Resemblyzer) — "does it sound like the same person?"
  2. MCD (Mel Cepstral Distortion)         — "how spectrally close is the clone?"

Usage:
    # Single comparison
    from similarity import compare
    result = compare("real.wav", "clone.wav")
    print(result)

    # Batch — all clones vs real voices
    from similarity import batch_report
    df = batch_report("input_voices/", "output_voices/")

    # CLI
    python similarity.py --real real.wav --clone clone.wav
    python similarity.py --batch --real_dir input_voices/ --clone_dir output_voices/
"""

import os, sys, json, argparse
import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────
SR       = 16000
N_MFCC   = 13
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# GE2E similarity thresholds
SIM_THRESHOLDS = {
    "excellent":  (0.90, "Same speaker — very high confidence"),
    "good":       (0.85, "Same speaker — high confidence"),
    "moderate":   (0.75, "Same speaker — moderate confidence"),
    "borderline": (0.70, "Borderline — may be same speaker"),
    "poor":       (0.00, "Different speaker"),
}

# MCD thresholds
MCD_THRESHOLDS = {
    "excellent": (0,   4,  "Near human quality"),
    "good":      (4,   8,  "Good TTS quality"),
    "moderate":  (8,   12, "Moderate quality"),
    "poor":      (12, 999, "Poor quality"),
}


# ══════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ══════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


# ══════════════════════════════════════════════════════════════════
# METRIC 1: GE2E SPEAKER SIMILARITY
# ══════════════════════════════════════════════════════════════════

_encoder = None  # cached

def get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder(device=DEVICE)
        except ImportError:
            print("  resemblyzer not installed — pip install resemblyzer")
            return None
    return _encoder


def ge2e_similarity(real_path: str, clone_path: str) -> dict:
    """
    GE2E speaker cosine similarity.
    Returns score 0-1, label, description.
    """
    enc = get_encoder()
    if enc is None:
        return {"score": None, "label": "unavailable",
                "description": "resemblyzer not installed"}

    try:
        from resemblyzer import preprocess_wav
        real_wav  = preprocess_wav(real_path)
        clone_wav = preprocess_wav(clone_path)

        e_real  = enc.embed_utterance(real_wav)
        e_clone = enc.embed_utterance(clone_wav)

        score = float(np.dot(e_real, e_clone) / (
            np.linalg.norm(e_real) * np.linalg.norm(e_clone)))

        # Get label
        label, description = "poor", "Different speaker"
        for lbl, (threshold, desc) in SIM_THRESHOLDS.items():
            if score >= threshold:
                label, description = lbl, desc
                break

        return {
            "score":       round(score, 4),
            "percent":     round(score * 100, 1),
            "label":       label,
            "description": description,
            "model":       "GE2E (Google, 2018)",
            "threshold":   "> 0.85 = same speaker",
        }
    except Exception as e:
        return {"score": None, "label": "error", "description": str(e)}


# ══════════════════════════════════════════════════════════════════
# METRIC 2: MCD (MEL CEPSTRAL DISTORTION)
# ══════════════════════════════════════════════════════════════════

def mcd_score(real_path: str, clone_path: str,
              sr: int = SR, n_mfcc: int = N_MFCC) -> dict:
    """
    Mel Cepstral Distortion — standard TTS quality metric.
    Lower = better. Unit: dB.
    """
    try:
        real  = load_audio(real_path,  sr)
        clone = load_audio(clone_path, sr)

        # Trim to minimum length
        min_len = min(len(real), len(clone))
        real    = real[:min_len]
        clone   = clone[:min_len]

        mfcc_real  = librosa.feature.mfcc(y=real,  sr=sr, n_mfcc=n_mfcc)
        mfcc_clone = librosa.feature.mfcc(y=clone, sr=sr, n_mfcc=n_mfcc)

        min_frames = min(mfcc_real.shape[1], mfcc_clone.shape[1])
        diff       = mfcc_real[:, :min_frames] - mfcc_clone[:, :min_frames]
        score      = float(np.mean(np.sqrt(2 * np.sum(diff**2, axis=0))))

        # Get label
        label, description = "poor", "Poor quality"
        for lbl, (lo, hi, desc) in MCD_THRESHOLDS.items():
            if lo <= score < hi:
                label, description = lbl, desc
                break

        return {
            "score":       round(score, 2),
            "unit":        "dB",
            "label":       label,
            "description": description,
            "note":        "Lower is better. Human resynthesis ~2-3 dB.",
        }
    except Exception as e:
        return {"score": None, "label": "error", "description": str(e)}


# ══════════════════════════════════════════════════════════════════
# COMBINED COMPARISON
# ══════════════════════════════════════════════════════════════════

def compare(real_path: str, clone_path: str,
            verbose: bool = True) -> dict:
    """
    Full similarity comparison — both metrics.
    This is the endpoint called by the demo UI.

    Returns:
    {
        "ge2e": {...},
        "mcd":  {...},
        "summary": "86.2% speaker similarity (GE2E) | MCD 7.3 dB",
        "verdict": "Good clone — recognizable as same person",
        "real_path": ...,
        "clone_path": ...
    }
    """
    ge2e = ge2e_similarity(real_path, clone_path)
    mcd  = mcd_score(real_path, clone_path)

    # Build summary
    ge2e_str = f"{ge2e['percent']}% speaker similarity" \
               if ge2e["score"] else "GE2E unavailable"
    mcd_str  = f"MCD {mcd['score']} dB" \
               if mcd["score"] else "MCD unavailable"
    summary  = f"{ge2e_str} | {mcd_str}"

    # Overall verdict
    ge2e_score = ge2e.get("score") or 0
    mcd_val    = mcd.get("score") or 999

    if ge2e_score >= 0.85 and mcd_val < 8:
        verdict = "High quality clone — very similar to real voice"
    elif ge2e_score >= 0.75:
        verdict = "Good clone — recognizable as same person"
    elif ge2e_score >= 0.65:
        verdict = "Moderate clone — some similarity detected"
    else:
        verdict = "Poor clone — speaker not convincingly reproduced"

    result = {
        "ge2e":       ge2e,
        "mcd":        mcd,
        "summary":    summary,
        "verdict":    verdict,
        "real_path":  real_path,
        "clone_path": clone_path,
    }

    if verbose:
        print(f"\n  Speaker Similarity Report")
        print(f"  {'─'*40}")
        print(f"  GE2E Score : {ge2e['percent']}%  ({ge2e['label']})")
        print(f"  Description: {ge2e['description']}")
        print(f"  MCD        : {mcd['score']} dB  ({mcd['label']})")
        print(f"  MCD note   : {mcd['description']}")
        print(f"  {'─'*40}")
        print(f"  Verdict    : {verdict}")

    return result


# ══════════════════════════════════════════════════════════════════
# BATCH REPORT — all clones vs real voices
# ══════════════════════════════════════════════════════════════════

def batch_report(real_dir: str = "input_voices",
                 clone_dir: str = "output_voices",
                 systems: list = None,
                 save_csv: bool = True) -> "pd.DataFrame":
    """
    Compute similarity for all clones vs real voices.
    Saves to analysis/similarity_report.csv
    Returns DataFrame.
    """
    import pandas as pd

    real_dir  = Path(real_dir)
    clone_dir = Path(clone_dir)
    systems   = systems or ["xtts", "yourtts", "chatterbox", "xtts_finetune"]

    # Build real voice lookup
    real_lookup = {}
    for wav in real_dir.glob("*_converted.wav"):
        name = wav.stem.lower()
        for suffix in ["_converted", "_vlsi", "_voice", "_test"]:
            name = name.replace(suffix, "")
        name = name.strip("_- ")
        real_lookup[name] = str(wav)

    print(f"  Real voices: {list(real_lookup.keys())}")
    print(f"  Systems: {systems}\n")

    rows = []
    print(f"  {'Person':<15} {'System':<15} {'Sent':<6} "
          f"{'GE2E%':>7} {'MCD(dB)':>8} {'Verdict'}")
    print(f"  {'─'*70}")

    for system in systems:
        sys_dir = clone_dir / system
        if not sys_dir.exists():
            print(f"  [skip] {system} — not found")
            continue

        for person_dir in sorted(sys_dir.iterdir()):
            if not person_dir.is_dir(): continue
            person = person_dir.name

            if person not in real_lookup:
                continue

            real_path = real_lookup[person]

            for wav in sorted(person_dir.glob("sent*.wav")):
                sent = wav.stem
                try:
                    ge2e = ge2e_similarity(real_path, str(wav))
                    mcd  = mcd_score(real_path, str(wav))

                    row = {
                        "person":      person,
                        "system":      system,
                        "sentence":    sent,
                        "ge2e_score":  ge2e.get("score"),
                        "ge2e_pct":    ge2e.get("percent"),
                        "ge2e_label":  ge2e.get("label"),
                        "mcd_score":   mcd.get("score"),
                        "mcd_label":   mcd.get("label"),
                        "real_path":   real_path,
                        "clone_path":  str(wav),
                    }
                    rows.append(row)

                    print(f"  {person:<15} {system:<15} {sent:<6} "
                          f"{ge2e.get('percent', 0):>6.1f}% "
                          f"{mcd.get('score', 0):>7.2f}  "
                          f"{ge2e.get('label','?')}")
                except Exception as e:
                    print(f"  FAIL {person} {system} {sent}: {e}")

    df = pd.DataFrame(rows)

    if save_csv and len(df) > 0:
        out = Path("analysis") / "similarity_report.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n  Saved: {out}")

        # Print summary by system
        print(f"\n  {'─'*50}")
        print(f"  SUMMARY BY SYSTEM")
        print(f"  {'─'*50}")
        print(f"  {'System':<20} {'GE2E Mean':>10} {'MCD Mean':>10}")
        print(f"  {'─'*50}")
        for sys in systems:
            sub = df[df["system"] == sys]
            if len(sub) == 0: continue
            ge2e_mean = sub["ge2e_score"].dropna().mean()
            mcd_mean  = sub["mcd_score"].dropna().mean()
            print(f"  {sys:<20} {ge2e_mean*100:>9.1f}% {mcd_mean:>9.2f} dB")

    return df


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Speaker Similarity Metrics")
    ap.add_argument("--real",      help="Real voice WAV file")
    ap.add_argument("--clone",     help="Clone WAV file")
    ap.add_argument("--batch",     action="store_true",
                    help="Run batch report on all clones")
    ap.add_argument("--real_dir",  default="input_voices")
    ap.add_argument("--clone_dir", default="output_voices")
    ap.add_argument("--json",      action="store_true",
                    help="Output JSON")
    args = ap.parse_args()

    if args.batch:
        df = batch_report(args.real_dir, args.clone_dir)

    elif args.real and args.clone:
        result = compare(args.real, args.clone)
        if args.json:
            print(json.dumps(result, indent=2))

    else:
        # Quick test
        print("Usage:")
        print("  python similarity.py --real input_voices/Abhishek_converted.wav "
              "--clone output_voices/xtts_finetune/abhishek/sent1.wav")
        print("  python similarity.py --batch")