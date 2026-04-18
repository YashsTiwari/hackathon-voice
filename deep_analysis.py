"""
deep_analysis.py — GPU-Accelerated Audio Forensics Analysis
══════════════════════════════════════════════════════════════════
PyTorch/Torchaudio optimized port. 
Spectral, Phase, Noise, and Temporal analysis run on CUDA.
Prosody (Praat) falls back to CPU workers.
"""

import warnings, os, json, time
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Setup & Device ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {DEVICE}")

BASE      = Path(__file__).parent
REAL_DIR  = BASE / "input_voices"
CLONE_DIR = BASE / "output_voices"
OUT_DIR   = BASE / "analysis"
PLOT_DIR  = OUT_DIR / "plots"

for d in [OUT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SR = 16000
SYSTEMS = ["xtts", "yourtts", "chatterbox", "xtts_finetune"]

# ── Pre-allocate GPU Transforms ───────────────────────────────────
# Doing this globally prevents massive memory allocation overhead in loops
N_FFT = 2048
HOP_LENGTH = 512

mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128, f_max=8000
).to(DEVICE)

mfcc_transform = T.MFCC(
    sample_rate=SR, n_mfcc=20, melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": 128}
).to(DEVICE)

lfcc_transform = T.LFCC(
    sample_rate=SR, n_lfcc=20, speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH}
).to(DEVICE)

centroid_transform = T.SpectralCentroid(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH
).to(DEVICE)

def compute_delta(tensor):
    """Compute delta features in PyTorch (equivalent to librosa.feature.delta)"""
    return T.ComputeDeltas()(tensor)

# ══════════════════════════════════════════════════════════════════
# AUDIO LOADING (GPU Bound)
# ══════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = SR) -> torch.Tensor:
    audio, orig_sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    if orig_sr != sr:
        resampler = T.Resample(orig_sr, sr).to(DEVICE)
        audio = resampler(audio.to(DEVICE))
    else:
        audio = audio.to(DEVICE)
        
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio / peak
        
    return audio.squeeze(0)  # Return 1D tensor on GPU

# ══════════════════════════════════════════════════════════════════
# BRANCH 1: SPECTRAL FEATURES (GPU)
# ══════════════════════════════════════════════════════════════════

def extract_spectral(audio: torch.Tensor, sr: int = SR) -> dict:
    feats = {}
    
    # ── Mel spectrogram ──
    mel = mel_transform(audio)
    mel_db = T.AmplitudeToDB()(mel)
    
    feats["mel_mean"] = mel_db.mean().item()
    feats["mel_std"]  = mel_db.std().item()
    
    # Skew and Kurtosis in PyTorch
    mel_flat = mel_db.flatten()
    mel_c = mel_flat - mel_flat.mean()
    m2 = torch.mean(mel_c**2)
    m3 = torch.mean(mel_c**3)
    m4 = torch.mean(mel_c**4)
    feats["mel_skew"]     = (m3 / (m2**(1.5) + 1e-10)).item()
    feats["mel_kurtosis"] = (m4 / (m2**2 + 1e-10) - 3.0).item()

    # ── STFT & Sub-band energy ratios ──
    window = torch.hann_window(N_FFT).to(DEVICE)
    stft_complex = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    stft_mag = torch.abs(stft_complex)
    
    freqs = torch.linspace(0, sr // 2, steps=stft_mag.shape[0], device=DEVICE)
    
    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not mask.any(): return 0.0
        return torch.mean(stft_mag[mask, :] ** 2).item()

    e_total = band_energy(0, sr//2) + 1e-10
    feats["energy_0_1k"]    = band_energy(0, 1000) / e_total
    feats["energy_1_4k"]    = band_energy(1000, 4000) / e_total
    feats["energy_4_8k"]    = band_energy(4000, 8000) / e_total
    feats["energy_8k_plus"] = band_energy(8000, sr//2) / e_total
    feats["hf_energy_ratio"]= feats["energy_4_8k"] + feats["energy_8k_plus"]

    # ── Spectral entropy ──
    def spectral_entropy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not mask.any(): return 0.0
        mag = stft_mag[mask, :].mean(dim=1)
        mag_norm = mag / (mag.sum() + 1e-10)
        return (-torch.sum(mag_norm * torch.log2(mag_norm + 1e-10))).item()

    feats["entropy_0_1k"]  = spectral_entropy(0, 1000)
    feats["entropy_1_4k"]  = spectral_entropy(1000, 4000)
    feats["entropy_4_8k"]  = spectral_entropy(4000, 8000)
    feats["entropy_total"] = spectral_entropy(0, sr//2)

    # ── Spectral Flux ──
    flux = torch.sqrt(torch.sum(torch.diff(stft_mag, dim=1) ** 2, dim=0))
    feats["spectral_flux_mean"] = flux.mean().item()
    feats["spectral_flux_std"]  = flux.std().item()

    # ── MFCC & LFCC ──
    mfcc = mfcc_transform(audio)
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = mfcc[i].mean().item()
        feats[f"mfcc{i+1}_std"]  = mfcc[i].std().item()
        
    mfcc_delta = compute_delta(mfcc)
    feats["mfcc_delta_mean"] = torch.abs(mfcc_delta).mean().item()
    feats["mfcc_delta_std"]  = mfcc_delta.std().item()

    lfcc = lfcc_transform(audio)
    feats["lfcc_mean"] = lfcc.mean().item()
    feats["lfcc_std"]  = lfcc.std().item()

    # ── RMS & Centroid ──
    centroid = centroid_transform(audio)
    feats["centroid_mean"] = centroid.mean().item()
    
    # RMS using PyTorch unfold
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    audio_unfolded = audio.unfold(0, frame_len, hop_len)
    rms = torch.sqrt(torch.mean(audio_unfolded**2, dim=1))
    feats["rms_mean"] = rms.mean().item()
    feats["rms_std"]  = rms.std().item()

    return feats

# ══════════════════════════════════════════════════════════════════
# BRANCH 2: PHASE ANALYSIS (GPU)
# ══════════════════════════════════════════════════════════════════

def extract_phase(audio: torch.Tensor, sr: int = SR) -> dict:
    feats = {}
    window = torch.hann_window(N_FFT).to(DEVICE)
    stft_complex = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    
    phase = torch.angle(stft_complex)
    magnitude = torch.abs(stft_complex)

    # Phase variance
    feats["phase_variance_mean"] = torch.var(phase, dim=1).mean().item()
    feats["phase_variance_std"]  = torch.var(phase, dim=1).std().item()

    # Group delay (approximated unwrapping using diff and wrap)
    phase_diff = torch.diff(phase, dim=0)
    # Wrap to [-pi, pi]
    phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi
    group_delay = -phase_diff
    
    feats["group_delay_mean"] = torch.abs(group_delay).mean().item()
    feats["group_delay_std"]  = group_delay.std().item()

    # Correlation
    mag_flat = magnitude.flatten()[:1000]
    phase_flat = torch.abs(phase).flatten()[:1000]
    if len(mag_flat) > 100:
        cov = torch.mean((mag_flat - mag_flat.mean()) * (phase_flat - phase_flat.mean()))
        std_m = mag_flat.std() + 1e-10
        std_p = phase_flat.std() + 1e-10
        feats["phase_mag_correlation"] = (cov / (std_m * std_p)).item()
    else:
        feats["phase_mag_correlation"] = 0.0

    return feats

# ══════════════════════════════════════════════════════════════════
# BRANCH 4: NOISE & TEMPORAL (GPU vectorization)
# ══════════════════════════════════════════════════════════════════

def extract_noise(audio: torch.Tensor, sr: int = SR) -> dict:
    feats = {}
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    
    audio_unfolded = audio.unfold(0, frame_len, hop_len)
    rms_frames = torch.sqrt(torch.mean(audio_unfolded**2, dim=1))
    threshold = rms_frames.mean() * 0.3
    
    voiced_mask = rms_frames > threshold
    silence_mask = ~voiced_mask

    feats["voiced_ratio"] = voiced_mask.float().mean().item()
    feats["silence_ratio"] = silence_mask.float().mean().item()

    if voiced_mask.any() and silence_mask.any():
        snr_voiced = rms_frames[voiced_mask].mean().item()
        snr_silence = rms_frames[silence_mask].mean().item()
        feats["snr_voiced"] = snr_voiced
        feats["snr_silence"] = snr_silence
        feats["snr_ratio"] = snr_voiced / (snr_silence + 1e-10)
        feats["snr_consistency"] = 1.0 / (feats["snr_ratio"] + 1e-10)
    else:
        feats["snr_voiced"] = feats["snr_silence"] = feats["snr_ratio"] = feats["snr_consistency"] = 0.0

    return feats

# ══════════════════════════════════════════════════════════════════
# PROSODY (CPU Fallback - Parselmouth required)
# ══════════════════════════════════════════════════════════════════

def extract_prosody_cpu(audio_np: np.ndarray, sr: int = SR) -> dict:
    """Keep exactly the same as your original script. Praat requires CPU arrays."""
    # ... (Insert original extract_prosody code here)
    # For brevity, returning empty dict here, but in your script, 
    # keep the exact parselmouth logic.
    return {"f0_mean": 0.0} # Placeholder to show where your original function sits

# ══════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION
# ══════════════════════════════════════════════════════════════════

def extract_all_features(audio_path: str) -> dict:
    """Combines GPU-accelerated Torch with CPU-bound Praat."""
    # 1. Load to GPU
    audio_gpu = load_audio(audio_path)
    
    feats = {}
    
    # 2. Run GPU tasks
    feats.update(extract_spectral(audio_gpu))
    feats.update(extract_phase(audio_gpu))
    feats.update(extract_noise(audio_gpu))
    
    # 3. Pull tensor back to CPU for Praat analysis
    audio_cpu = audio_gpu.cpu().numpy()
    feats.update(extract_prosody_cpu(audio_cpu))
    
    return feats

# Replace your original process_one with this:
def process_one(entry):
    try:
        # Wrap everything in torch.no_grad() to save massive amounts of VRAM
        with torch.no_grad():
            feats = extract_all_features(entry["filepath"])
            print(f"  OK: {entry['person']:12s} | {entry['system']:15s} | {entry['sentence']}")
            return {**entry, **feats}
    except Exception as e:
        print(f"  FAIL: {entry['person']} {entry['system']}: {e}")
        return None

# ... Keep your existing find_all_files, analyze_discrimination, and main() blocks.
def analyze_discrimination(df: pd.DataFrame) -> dict:
    """
    For each feature, compute:
    - Mean for real vs each fake system
    - Effect size (Cohen's d)
    - Consistency across speakers
    Returns ranked feature list with hardcode thresholds.
    """
    feature_cols = [c for c in df.columns
                    if c not in ["person","system","sentence","label","filepath"]]

    results = {}
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    for feat in feature_cols:
        real_vals = real_df[feat].dropna().values
        fake_vals = fake_df[feat].dropna().values

        if len(real_vals) < 3 or len(fake_vals) < 3:
            continue

        real_mean = np.mean(real_vals)
        fake_mean = np.mean(fake_vals)
        real_std  = np.std(real_vals)
        fake_std  = np.std(fake_vals)

        # Cohen's d — effect size
        pooled_std = np.sqrt((real_std**2 + fake_std**2) / 2) + 1e-10
        cohens_d   = abs(real_mean - fake_mean) / pooled_std

        # Consistency: does this feature work across all fake systems?
        per_system = {}
        for sys in df[df["label"]=="fake"]["system"].unique():
            sys_vals = df[(df["label"]=="fake") & (df["system"]==sys)][feat].dropna().values
            if len(sys_vals) > 0:
                per_system[sys] = float(np.mean(sys_vals))

        system_consistency = 1.0 - np.std(list(per_system.values())) / (
            abs(np.mean(list(per_system.values()))) + 1e-10
        ) if per_system else 0.0

        # Suggest threshold
        # Use midpoint between real and fake means as threshold
        threshold = (real_mean + fake_mean) / 2
        direction = ">" if fake_mean > real_mean else "<"

        results[feat] = {
            "real_mean":    float(real_mean),
            "real_std":     float(real_std),
            "fake_mean":    float(fake_mean),
            "fake_std":     float(fake_std),
            "cohens_d":     float(cohens_d),
            "consistency":  float(system_consistency),
            "threshold":    float(threshold),
            "direction":    direction,
            "score":        float(cohens_d * max(0, system_consistency)),
        }

    return dict(sorted(results.items(),
                       key=lambda x: x[1]["score"], reverse=True))


def find_all_files() -> list[dict]:
    """Find all real and fake audio files."""
    entries = []

    # Build real voice lookup: person_name → filepath
    real_lookup = {}
    for wav in REAL_DIR.glob("*_converted.wav"):
        name = wav.stem.lower()
        for suffix in ["_converted","_vlsi","_voice","_test","_audio"]:
            name = name.replace(suffix, "")
        name = name.strip("_- ")
        real_lookup[name] = str(wav)

    print(f"  Real voices found: {list(real_lookup.keys())}")

    # Fake clones
    for system in SYSTEMS:
        sys_dir = CLONE_DIR / system
        if not sys_dir.exists():
            print(f"  [skip] {system} — folder not found")
            continue
        for person_dir in sorted(sys_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            wavs = list(person_dir.glob("sent*.wav"))
            if not wavs:
                continue
            for wav in sorted(wavs):
                sent = wav.stem
                entries.append({
                    "person":   person,
                    "system":   system,
                    "sentence": sent,
                    "label":    "fake",
                    "filepath": str(wav),
                })
            # Add matching real voice once per person
            if person in real_lookup:
                real_already = any(
                    e["person"] == person and e["system"] == "real"
                    for e in entries
                )
                if not real_already:
                    entries.append({
                        "person":   person,
                        "system":   "real",
                        "sentence": "full",
                        "label":    "real",
                        "filepath": real_lookup[person],
                    })
            else:
                print(f"  [warn] No real voice found for: {person}")

    print(f"  Total entries: {len(entries)}")
    real_count = sum(1 for e in entries if e["label"] == "real")
    fake_count = sum(1 for e in entries if e["label"] == "fake")
    print(f"  Real: {real_count}  Fake: {fake_count}")
    return entries

# ══════════════════════════════════════════════════════════════════
# VISUALIZATION (Adapted for Torch Tensors)
# ══════════════════════════════════════════════════════════════════

def plot_comparison(real_audio_tensor, fake_audio_tensor, person, system, sent, sr=SR):
    """Side-by-side spectrogram + pitch comparison. Casts Torch tensors back to NumPy for plotting."""
    
    # Cast GPU tensors back to CPU NumPy arrays for Librosa/Matplotlib
    if isinstance(real_audio_tensor, torch.Tensor):
        real_audio = real_audio_tensor.cpu().numpy()
    else:
        real_audio = real_audio_tensor
        
    if isinstance(fake_audio_tensor, torch.Tensor):
        fake_audio = fake_audio_tensor.cpu().numpy()
    else:
        fake_audio = fake_audio_tensor

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    audios = [("Real", real_audio), (f"Fake ({system})", fake_audio)]

    for col, (label, audio) in enumerate(audios):
        # Mel spectrogram
        ax1 = fig.add_subplot(gs[0, col])
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                  x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f"{label} — Mel Spectrogram", fontsize=10)
        plt.colorbar(ax1.images[0], ax=ax1, format='%+2.0f dB')

        # Pitch contour
        ax2 = fig.add_subplot(gs[1, col])
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        times = librosa.times_like(pitch, sr=sr)
        pitch_voiced = np.where(pitch > 0, pitch, np.nan)
        ax2.plot(times, pitch_voiced, color='steelblue' if col==0 else 'coral',
                 linewidth=0.8, alpha=0.8)
        ax2.set_ylabel("F0 (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(f"{label} — Pitch Contour", fontsize=10)
        ax2.set_ylim(0, 500)

        # RMS energy
        ax3 = fig.add_subplot(gs[2, col])
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr,
                                        hop_length=256)
        ax3.plot(t_rms, rms, color='green' if col==0 else 'orange',
                 linewidth=0.8)
        ax3.set_ylabel("RMS Energy")
        ax3.set_xlabel("Time (s)")
        ax3.set_title(f"{label} — Energy Profile", fontsize=10)

    plt.suptitle(f"{person} | {system} | {sent}", fontsize=12, fontweight='bold')
    out = PLOT_DIR / f"{person}_{system}_{sent}.png"
    plt.savefig(str(out), dpi=100, bbox_inches='tight')
    plt.close()
    return str(out)


# ══════════════════════════════════════════════════════════════════
# DISCRIMINATION REPORT
# ══════════════════════════════════════════════════════════════════

def print_discrimination_report(disc: dict, top_n: int = 30) -> tuple:
    """Generates the text report and extracts hardcode thresholds."""
    lines = [
        "=" * 80,
        "  FEATURE DISCRIMINATION REPORT",
        "  Real vs Fake — Cohen's d effect size + consistency across systems",
        "=" * 80,
        "",
        f"{'Feature':<40} {'Real':>8} {'Fake':>8} {'Cohen_d':>8} {'Consist':>8} {'Threshold':>12}",
        "─" * 80,
    ]

    hardcode = {}
    for feat, vals in list(disc.items())[:top_n]:
        hardcode_flag = "★ HARDCODE" if vals["cohens_d"] > 1.0 and vals["consistency"] > 0.5 else ""
        lines.append(
            f"{feat:<40} "
            f"{vals['real_mean']:>8.3f} "
            f"{vals['fake_mean']:>8.3f} "
            f"{vals['cohens_d']:>8.3f} "
            f"{vals['consistency']:>8.3f} "
            f"{vals['direction']}{vals['threshold']:>11.3f}  {hardcode_flag}"
        )
        if vals["cohens_d"] > 1.0 and vals["consistency"] > 0.5:
            hardcode[feat] = {
                "threshold": vals["threshold"],
                "direction": vals["direction"],
                "real_mean": vals["real_mean"],
                "fake_mean": vals["fake_mean"],
                "cohens_d":  vals["cohens_d"],
            }

    lines += [
        "",
        f"  Features suitable for hardcoding (Cohen's d > 1.0, consistency > 0.5):",
        f"  {len(hardcode)} features identified",
        "",
    ]
    for feat, vals in hardcode.items():
        lines.append(f"  {feat}: {vals['direction']}{vals['threshold']:.4f}  "
                    f"(real={vals['real_mean']:.3f}, fake={vals['fake_mean']:.3f}, "
                    f"d={vals['cohens_d']:.2f})")

    lines.append("=" * 80)
    return "\n".join(lines), hardcode

def main():
    print("\n" + "="*70)
    print("  DEEP AUDIO FORENSICS ANALYSIS")
    print(f"  Output: {OUT_DIR}")
    print("="*70)

    files = find_all_files()
    print(f"\n  Found {len(files)} files to analyze")

    systems = {}
    for e in files:
        systems[e["system"]] = systems.get(e["system"], 0) + 1
    for sys, count in sorted(systems.items()):
        print(f"    {sys:20s}: {count} files")

    # Check for existing results
    csv_path = OUT_DIR / "features.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        done_paths = set(existing["filepath"].tolist())
        files = [f for f in files if f["filepath"] not in done_paths]
        print(f"\n  Resuming — {len(files)} remaining")
        all_rows = existing.to_dict("records")
    else:
        all_rows = []

    # Extract features — parallel
    t_start = time.time()
    n_jobs  = 16

    def process_one(entry):
        try:
            if entry["label"] == "real":
                audio_full = load_audio(entry["filepath"])
                chunk = int(8 * SR)
                mid   = len(audio_full) // 2
                clips = [audio_full[SR*10:SR*10+chunk],
                         audio_full[mid:mid+chunk],
                         audio_full[-chunk-SR*5:-SR*5]]
                clips = [c for c in clips if len(c) == chunk]
                if clips:
                    all_feats = [extract_all_features(c)
                                 for c in clips]
                    feats = {k: float(np.mean([f[k] for f in all_feats]))
                             for k in all_feats[0]}
                else:
                    feats = extract_all_features(entry["filepath"])
            else:
                feats = extract_all_features(entry["filepath"])
            print(f"  OK: {entry['person']:12s} | {entry['system']:15s} | {entry['sentence']}")
            return {**entry, **feats}
        except Exception as e:
            print(f"  FAIL: {entry['person']} {entry['system']}: {e}")
            return None

    print(f"  Running {n_jobs} parallel workers on {len(files)} files...")
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(process_one)(entry) for entry in files
    )
    all_rows.extend([r for r in results if r is not None])
    elapsed = (time.time() - t_start) / 60
    print(f"  Done: {len(all_rows)} files in {elapsed:.1f} min")

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Features saved: {csv_path}")
    print(f"  Total rows: {len(df)}")

    # Generate comparison plots (first person only to save time)
    print("\n  Generating comparison plots...")
    xtts_dir = CLONE_DIR / "xtts"
    if xtts_dir.exists():
        first_person = sorted(xtts_dir.iterdir())[0].name
        real_entry   = next((r for r in all_rows
                             if r["person"] == first_person
                             and r["system"] == "real"), None)
        if real_entry:
            real_audio = load_audio(real_entry["filepath"])
            # Trim to 10 seconds for comparison
            real_audio = real_audio[:SR*10]
            for system in ["xtts", "yourtts", "chatterbox"]:
                fake_entry = next((r for r in all_rows
                                   if r["person"] == first_person
                                   and r["system"] == system
                                   and r["sentence"] == "sent1"), None)
                if fake_entry:
                    fake_audio = load_audio(fake_entry["filepath"])
                    plot_comparison(real_audio, fake_audio,
                                    first_person, system, "sent1")
                    print(f"  Plot: {first_person} vs {system}")

    # Discrimination analysis
    print("\n  Running discrimination analysis...")
    disc      = analyze_discrimination(df)
    report, hardcode = print_discrimination_report(disc)
    print("\n" + report)

    # Save report
    report_path = OUT_DIR / "discrimination.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save hardcode thresholds
    thresh_path = OUT_DIR / "hardcode_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(hardcode, f, indent=2)
    print(f"\n  Discrimination report: {report_path}")
    print(f"  Hardcode thresholds:   {thresh_path}")
    print(f"\n  DONE. Top discriminating features saved.")


if __name__ == "__main__":
    main()
