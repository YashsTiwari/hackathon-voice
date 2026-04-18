"""
detector.py — Voice Deepfake Detector
══════════════════════════════════════════════════════════════════════
4-layer ensemble detector:
  Layer 1: Hardcoded thresholds (insurance — universal signals)
  Layer 2: XGBoost on 138 hand-crafted features (trained on our data)
  Layer 3: AASIST pretrained (state-of-art neural anti-spoofing)
  Layer 4: Watermark check (catches our own cloner with ~100% certainty)

Final score = weighted ensemble of all layers
Watermark detection overrides everything → FAKE if found

Usage:
    from detector import VoiceDetector
    det = VoiceDetector()
    result = det.predict("audio.wav")
    print(result)  # {'label': 'fake', 'confidence': 0.94, 'signals': [...]}

    # Or from command line:
    python detector.py --audio path/to/audio.wav
"""

import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
DETECTOR_DIR   = Path(__file__).parent
MODEL_DIR      = DETECTOR_DIR / "detector_models"
THRESHOLD_FILE = DETECTOR_DIR / "analysis" / "hardcode_thresholds.json"
XGBOOST_MODEL  = MODEL_DIR / "xgboost_detector.pkl"
AASIST_DIR     = DETECTOR_DIR / "aasist"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

# ── Watermark config ───────────────────────────────────────────────
WATERMARK_FREQ      = 7823   # Hz — injected into our clones
WATERMARK_AMPLITUDE = 0.0003
WATERMARK_THRESHOLD = 0.0002


# ══════════════════════════════════════════════════════════════════
# LAYER 1: HARDCODED THRESHOLDS
# These are filled from analysis/hardcode_thresholds.json
# Defaults below are educated guesses — replaced after analysis runs
# ══════════════════════════════════════════════════════════════════

# Default thresholds (placeholders — will be overridden from JSON)
DEFAULT_THRESHOLDS = {
    # Tier 1 — near certain (any single one = high confidence fake)
    "breathing_count": {
        "direction": "<", "threshold": 1.0,
        "weight": 0.9, "tier": 1,
        "description": "No breathing sounds detected"
    },
    "snr_ratio": {
        "direction": ">", "threshold": 3.0,
        "weight": 0.85, "tier": 1,
        "description": "Dead silence in pauses (no room noise)"
    },
    "f0_jitter_local": {
        "direction": "<", "threshold": 0.01,
        "weight": 0.85, "tier": 1,
        "description": "F0 too smooth — unnatural pitch stability"
    },

    # Tier 2 — strong (2+ needed for high confidence)
    "hf_energy_ratio": {
        "direction": ">", "threshold": 0.22,
        "weight": 0.7, "tier": 2,
        "description": "High-frequency aliasing (vocoder artifact)"
    },
    "shimmer": {
        "direction": "<", "threshold": 0.02,
        "weight": 0.65, "tier": 2,
        "description": "Shimmer too low — unnatural amplitude stability"
    },
    "pause_cv": {
        "direction": "<", "threshold": 0.20,
        "weight": 0.65, "tier": 2,
        "description": "Pause durations too uniform (metronomic)"
    },
    "spectral_flatness_mean": {
        "direction": ">", "threshold": 0.35,
        "weight": 0.60, "tier": 2,
        "description": "Spectrum too flat — over-smoothed by vocoder"
    },
    "noise_floor_cv": {
        "direction": "<", "threshold": 0.15,
        "weight": 0.60, "tier": 2,
        "description": "Noise floor too consistent — generated silence"
    },

    # Tier 3 — weak signals (3+ in combination)
    "respiratory_am_energy": {
        "direction": "<", "threshold": 0.001,
        "weight": 0.40, "tier": 3,
        "description": "No respiratory amplitude modulation"
    },
    "dct_periodic_peak_score": {
        "direction": ">", "threshold": 3.0,
        "weight": 0.40, "tier": 3,
        "description": "Periodic DCT peaks (upsampling artifacts)"
    },
    "waveform_kurtosis": {
        "direction": "<", "threshold": 2.5,
        "weight": 0.35, "tier": 3,
        "description": "Low waveform kurtosis — too Gaussian"
    },
    "group_delay_jag": {
        "direction": "<", "threshold": 0.1,
        "weight": 0.35, "tier": 3,
        "description": "Group delay too smooth"
    },
    "f0_complexity": {
        "direction": "<", "threshold": 10.0,
        "weight": 0.35, "tier": 3,
        "description": "Pitch contour too simple"
    },
}


def load_thresholds() -> dict:
    """Load actual thresholds from analysis output, fall back to defaults."""
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE) as f:
            analysis_thresholds = json.load(f)

        # Merge analysis results into defaults
        thresholds = DEFAULT_THRESHOLDS.copy()
        for feat, vals in analysis_thresholds.items():
            if feat in thresholds:
                thresholds[feat]["threshold"] = vals["threshold"]
                thresholds[feat]["direction"] = vals["direction"]
            # Add new features from analysis
            elif vals.get("cohens_d", 0) > 0.8:
                thresholds[feat] = {
                    "direction":   vals["direction"],
                    "threshold":   vals["threshold"],
                    "weight":      min(0.5, vals["cohens_d"] / 10),
                    "tier":        3,
                    "description": f"Feature: {feat}"
                }
        print(f"  Loaded {len(analysis_thresholds)} thresholds from analysis")
        return thresholds
    else:
        print("  Using default thresholds (run analysis first for better accuracy)")
        return DEFAULT_THRESHOLDS


def hardcoded_check(features: dict, thresholds: dict) -> tuple[float, list[str]]:
    """
    Layer 1: Rule-based threshold check.
    Returns (score 0-1, list of triggered signals)
    """
    triggered = []
    tier1_hits = 0
    tier2_hits = 0
    tier3_hits = 0
    weighted_sum = 0.0
    weight_total = 0.0

    for feat, config in thresholds.items():
        val = features.get(feat)
        if val is None:
            continue

        direction = config["direction"]
        threshold = config["threshold"]
        fired     = (direction == ">" and val > threshold) or \
                    (direction == "<" and val < threshold)

        if fired:
            triggered.append(f"{feat} ({config['description']})")
            tier = config.get("tier", 3)
            if tier == 1: tier1_hits += 1
            elif tier == 2: tier2_hits += 1
            else: tier3_hits += 1
            weighted_sum += config["weight"]

        weight_total += config["weight"]

    # Score based on tier hits
    if tier1_hits >= 1:
        base_score = 0.85 + min(0.14, tier1_hits * 0.05)
    elif tier2_hits >= 2:
        base_score = 0.70 + min(0.14, tier2_hits * 0.05)
    elif tier2_hits >= 1 and tier3_hits >= 2:
        base_score = 0.60
    elif tier3_hits >= 3:
        base_score = 0.55
    else:
        base_score = weighted_sum / (weight_total + 1e-10) * 0.5

    return float(base_score), triggered


# ══════════════════════════════════════════════════════════════════
# LAYER 2: XGBOOST ON HAND-CRAFTED FEATURES
# ══════════════════════════════════════════════════════════════════

def load_xgboost():
    """Load trained XGBoost model."""
    if not XGBOOST_MODEL.exists():
        return None
    import pickle
    with open(XGBOOST_MODEL, "rb") as f:
        return pickle.load(f)


def xgboost_predict(features: dict, model) -> float:
    """
    Layer 2: XGBoost prediction.
    Returns probability of being fake (0-1).
    """
    if model is None:
        return 0.5  # neutral if not trained yet

    import pandas as pd
    # Get feature columns in same order as training
    feature_cols = model.get_booster().feature_names
    row = {col: features.get(col, 0.0) for col in feature_cols}
    df  = pd.DataFrame([row])
    prob = model.predict_proba(df)[0][1]  # probability of class 1 (fake)
    return float(prob)


def train_xgboost(features_csv: str, save_path: str = None):
    """
    Train XGBoost on extracted features.
    Call this after deep_analysis.py finishes.
    """
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    import pickle

    print("\n  Training XGBoost detector...")
    df = pd.read_csv(features_csv)

    # Label encoding
    df["label_int"] = (df["label"] == "fake").astype(int)

    # Feature columns
    meta_cols = ["person", "system", "sentence", "label",
                 "filepath", "label_int"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].fillna(0)
    y = df["label_int"]

    print(f"  Dataset: {len(df)} samples, {len(feat_cols)} features")
    print(f"  Real: {(y==0).sum()}  Fake: {(y==1).sum()}")

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train XGBoost
    clf = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    # Evaluate
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n  AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["real", "fake"]))

    # Feature importance
    importance = sorted(
        zip(feat_cols, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\n  Top 15 features by importance:")
    for feat, imp in importance[:15]:
        print(f"    {feat:40s}: {imp:.4f}")

    # Save
    save_path = save_path or str(XGBOOST_MODEL)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n  Model saved: {save_path}")
    return clf, auc


# ══════════════════════════════════════════════════════════════════
# LAYER 3: AASIST PRETRAINED
# ══════════════════════════════════════════════════════════════════

def load_aasist():
    """Load pretrained AASIST model."""
    aasist_model_path = AASIST_DIR / "models" / "weights" / "AASIST-L.pth"
    aasist_config     = AASIST_DIR / "config" / "AASIST-L.conf"

    if not aasist_model_path.exists():
        print("  AASIST model not found — skipping (run setup_aasist.py)")
        return None

    try:
        sys.path.insert(0, str(AASIST_DIR))
        import torch
        from models.AASIST import Model as AASISTModel

        with open(aasist_config) as f:
            import yaml
            config = yaml.safe_load(f)

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        model  = AASISTModel(config["model_config"]).to(device)
        model.load_state_dict(
            __import__("torch").load(aasist_model_path,
                                     map_location=device))
        model.eval()
        print(f"  AASIST loaded on {device}")
        return model
    except Exception as e:
        print(f"  AASIST load failed: {e}")
        return None


def aasist_predict(audio: np.ndarray, model, sr: int = 16000) -> float:
    """
    Layer 3: AASIST prediction.
    Returns probability of being fake (0-1).
    """
    if model is None:
        return 0.5  # neutral fallback

    try:
        import torch
        device = next(model.parameters()).device

        # AASIST expects specific input format
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            _, output = model(audio_tensor)
            # AASIST output: [batch, 2] — [real_score, fake_score]
            scores = torch.softmax(output, dim=1)
            fake_prob = scores[0][1].item()

        return float(fake_prob)
    except Exception as e:
        print(f"  AASIST inference error: {e}")
        return 0.5


# ══════════════════════════════════════════════════════════════════
# LAYER 4: WATERMARK DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_watermark(audio: np.ndarray, sr: int = SR) -> tuple[bool, float]:
    """
    ~7823Hz tone at amplitude 0.0003.
    """
    fft     = np.abs(np.fft.rfft(audio))
    freqs   = np.fft.rfftfreq(len(audio), 1/sr)
    idx     = np.argmin(np.abs(freqs - WATERMARK_FREQ))

    # Energy in narrow band around marker frequency (±5 bins)
    band_energy   = np.mean(fft[max(0,idx-5):idx+6])
    # Compare to nearby background
    bg_energy     = np.mean(np.concatenate([
        fft[max(0,idx-50):max(0,idx-10)],
        fft[idx+10:idx+50]
    ]))
    snr           = band_energy / (bg_energy + 1e-10)
    found         = snr > 3.0  # 3x above background = watermark present
    confidence    = min(1.0, (snr - 1.0) / 5.0) if snr > 1.0 else 0.0

    return found, float(confidence)


def inject_watermark(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    t      = np.arange(len(audio)) / sr
    marker = WATERMARK_AMPLITUDE * np.sin(2 * np.pi * WATERMARK_FREQ * t)
    return audio + marker.astype(audio.dtype)


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (calls deep_analysis functions)
# ══════════════════════════════════════════════════════════════════

def extract_features_for_inference(audio: np.ndarray, sr: int = SR) -> dict:
    """
    Extract all features for inference.
    Bridges the gap between NumPy-based loading and GPU-based PyTorch extraction.
    """
    try:
        sys.path.insert(0, str(DETECTOR_DIR))
        import deep_analysis as da
        import torch

        # 1. Convert the NumPy array to a PyTorch Tensor and push to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_tensor = torch.from_numpy(audio).to(device)

        feats = {}

        # 2. Pass the TENSOR to the PyTorch-accelerated functions
        feats.update(da.extract_spectral(audio_tensor))
        feats.update(da.extract_phase(audio_tensor))
        feats.update(da.extract_noise(audio_tensor))

        # 3. Pass the original NUMPY ARRAY to the CPU-bound functions
        # (Parselmouth/Praat will crash if you feed it a GPU Tensor)
        feats.update(da.extract_prosody_cpu(audio))
        
        if hasattr(da, 'extract_temporal'):
            feats.update(da.extract_temporal(audio))
            
        if hasattr(da, 'extract_crossdomain'):
            feats.update(da.extract_crossdomain(audio))

        return feats
        
    except Exception as e:
        print(f"  Feature extraction error: {e}")
        import traceback
        traceback.print_exc()  # This will print exactly where it fails if it happens again
        return {}


def load_audio_for_inference(path: str, sr: int = SR,
                               max_duration: float = 30.0) -> np.ndarray:
    """Load and preprocess audio for inference."""
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Cap at max_duration
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        # Take middle section
        mid   = len(audio) // 2
        start = max(0, mid - max_samples // 2)
        audio = audio[start:start + max_samples]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio

# ══════════════════════════════════════════════════════════════════
# MAIN DETECTOR CLASS
# ══════════════════════════════════════════════════════════════════

class VoiceDetector:
    """
    Main detector class. Loads all layers and runs ensemble prediction.
    """

    # Ensemble weights — tunable
    WEIGHTS = {
        "hardcoded": 0.10,
        "xgboost":   0.65,
        "aasist":    0.20,
        "watermark": 0.05,  # small weight — overrides via flag instead
    }

    def __init__(self, verbose: bool = True):
        self.verbose    = verbose
        self.thresholds = load_thresholds()
        self.xgb_model  = load_xgboost()
        self.aasist     = load_aasist()

        if verbose:
            print(f"\n  VoiceDetector initialized:")
            print(f"  Thresholds: {len(self.thresholds)} rules")
            print(f"  XGBoost:    {'loaded' if self.xgb_model else 'not trained yet'}")
            print(f"  AASIST:     {'loaded' if self.aasist else 'not available'}")

    def predict(self, audio_path: str) -> dict:
        """
        Full prediction pipeline.
        Returns dict with label, confidence, per-layer scores, signals.
        """
        t0 = time.time()

        # Load audio
        audio = load_audio_for_inference(audio_path)

        # ── Layer 4: Watermark check (fast, do first) ──────────────
        wm_found, wm_conf = detect_watermark(audio)
        if wm_found:
            return {
                "label":      "fake",
                "confidence": float(0.99),
                "verdict":    "FAKE — our watermark detected",
                "signals":    ["watermark_detected"],
                "scores": {
                    "watermark": float(wm_conf),
                    "hardcoded": None,
                    "xgboost":   None,
                    "aasist":    None,
                },
                "elapsed": time.time() - t0,
            }

        # ── Extract features (shared by layers 1 and 2) ────────────
        features = extract_features_for_inference(audio)

        # ── Layer 1: Hardcoded thresholds ─────────────────────────
        hc_score, signals = hardcoded_check(features, self.thresholds)

        # ── Layer 2: XGBoost ──────────────────────────────────────
        xgb_score = xgboost_predict(features, self.xgb_model)

        # ── Layer 3: AASIST ───────────────────────────────────────
        aasist_score = aasist_predict(audio, self.aasist)

        # ── Ensemble ──────────────────────────────────────────────
        # Adjust weights based on what's available
        w = self.WEIGHTS.copy()
        if self.xgb_model is None:
            w["hardcoded"] += w["xgboost"]
            w["xgboost"]   = 0
        if self.aasist is None:
            w["hardcoded"] += w["aasist"] * 0.5
            w["xgboost"]   += w["aasist"] * 0.5
            w["aasist"]     = 0

        total_w = w["hardcoded"] + w["xgboost"] + w["aasist"]
        ensemble_score = (
            w["hardcoded"] * hc_score +
            w["xgboost"]   * xgb_score +
            w["aasist"]    * aasist_score
        ) / (total_w + 1e-10)

        label      = "fake" if ensemble_score > 0.5 else "real"
        confidence = ensemble_score if label == "fake" else 1 - ensemble_score

        # Build verdict string
        if label == "fake":
            if ensemble_score > 0.85:
                verdict = "FAKE — high confidence"
            elif ensemble_score > 0.65:
                verdict = "FAKE — moderate confidence"
            else:
                verdict = "LIKELY FAKE — low confidence"
        else:
            if ensemble_score < 0.25:
                verdict = "REAL — high confidence"
            elif ensemble_score < 0.40:
                verdict = "REAL — moderate confidence"
            else:
                verdict = "LIKELY REAL — low confidence"

        return {
            "label":      label,
            "confidence": float(confidence),
            "verdict":    verdict,
            "signals":    signals,
            "scores": {
                "ensemble":  float(ensemble_score),
                "hardcoded": float(hc_score),
                "xgboost":   float(xgb_score),
                "aasist":    float(aasist_score),
                "watermark": 0.0,
            },
            "top_features": {
                k: float(features.get(k, 0))
                for k in ["breathing_count", "snr_ratio", "f0_jitter_local",
                          "hf_energy_ratio", "shimmer", "pause_cv"]
            },
            "elapsed": float(time.time() - t0),
        }

    def predict_batch(self, audio_paths: list[str],
                       n_jobs: int = 4) -> list[dict]:
        """Predict on multiple files in parallel."""
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs)(
            delayed(self.predict)(p) for p in audio_paths
        )


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Voice Deepfake Detector")
    sub    = parser.add_subparsers(dest="command")

    # predict
    p_pred = sub.add_parser("predict", help="Predict on audio file(s)")
    p_pred.add_argument("audio", nargs="+", help="Audio file path(s)")
    p_pred.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable")

    # train
    p_train = sub.add_parser("train", help="Train XGBoost on features.csv")
    p_train.add_argument("--features", default="analysis/features.csv")
    p_train.add_argument("--out",      default=str(XGBOOST_MODEL))

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate detector on test set")
    p_eval.add_argument("--features", default="analysis/features.csv")

    args = parser.parse_args()

    if args.command == "train":
        clf, auc = train_xgboost(args.features, args.out)
        print(f"\n  Final AUC: {auc:.4f}")

    elif args.command == "predict":
        det = VoiceDetector()
        for audio_path in args.audio:
            print(f"\n  Analyzing: {audio_path}")
            result = det.predict(audio_path)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"  Verdict:    {result['verdict']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  Scores:")
                for k, v in result["scores"].items():
                    if v is not None:
                        print(f"    {k:12s}: {v:.3f}")
                if result["signals"]:
                    print(f"  Signals triggered:")
                    for s in result["signals"][:5]:
                        print(f"    • {s}")
                print(f"  Time: {result['elapsed']:.1f}s")

    elif args.command == "evaluate":
        import pandas as pd
        from sklearn.metrics import roc_auc_score, classification_report

        det = VoiceDetector()
        df  = pd.read_csv(args.features)

        print(f"\n  Evaluating on {len(df)} samples...")
        preds, labels = [], []

        for _, row in df.iterrows():
            if not Path(row["filepath"]).exists():
                continue
            result = det.predict(row["filepath"])
            preds.append(result["scores"]["ensemble"])
            labels.append(1 if row["label"] == "fake" else 0)

        preds  = np.array(preds)
        labels = np.array(labels)
        binary = (preds > 0.5).astype(int)

        print(f"\n  AUC: {roc_auc_score(labels, preds):.4f}")
        print(classification_report(labels, binary,
                                     target_names=["real","fake"]))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
