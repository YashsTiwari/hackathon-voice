"""
prepare_dataset.py — replacement for XTTS formatter
Uses openai-whisper + soundfile instead of faster-whisper
No ffmpeg needed, works with CUDA 12.1

Usage:
    python prepare_dataset.py --wav path/to/voice.wav --out_dir path/to/dataset
"""

import argparse, os, time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import whisper
from pathlib import Path

def prepare(wav_path: str, out_dir: str, language: str = "en",
            eval_pct: float = 0.15, speaker: str = "speaker"):

    out_dir = Path(out_dir)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "metadata_train.csv"
    eval_csv  = out_dir / "metadata_eval.csv"

    if train_csv.exists() and eval_csv.exists():
        df = pd.read_csv(train_csv, sep="|")
        print(f"  [skip] Already done — {len(df)} train samples")
        return str(train_csv), str(eval_csv)

    # Load audio
    print(f"  Loading audio: {wav_path}")
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    total_dur = len(audio) / sr
    print(f"  Duration: {total_dur:.1f}s")

    # Load whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Whisper small on {device}...")
    model = whisper.load_model("small", device=device)

    # Transcribe with word timestamps
    print(f"  Transcribing...")
    t0 = time.time()
    result = model.transcribe(
        audio,
        language        = language,
        word_timestamps = True,
        verbose         = False,
    )
    print(f"  Transcribed in {time.time()-t0:.1f}s")

    # Build sentence-level chunks from word timestamps
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    chunk_idx = 0
    buffer    = 0.2  # seconds of padding

    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue

        # Get start/end with buffer
        start = max(0, seg["start"] - buffer)
        end   = min(total_dur, seg["end"] + buffer)
        dur   = end - start

        # Skip too short or too long
        if dur < 1.0 or dur > 12.0:
            continue

        # Extract audio chunk
        s_idx = int(start * sr)
        e_idx = int(end   * sr)
        chunk = audio[s_idx:e_idx]

        # Save chunk
        rel_path  = f"wavs/chunk_{chunk_idx:06d}.wav"
        abs_path  = out_dir / rel_path
        sf.write(str(abs_path), chunk, sr)

        from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
        text_clean = multilingual_cleaners(text, language)

        metadata["audio_file"].append(rel_path)
        metadata["text"].append(text_clean)
        metadata["speaker_name"].append(speaker)
        chunk_idx += 1

    print(f"  Created {chunk_idx} chunks")

    if chunk_idx < 5:
        print("  WARNING: Very few chunks — audio may be too short or quiet")

    # Split train/eval
    df = pd.DataFrame(metadata).sample(frac=1, random_state=42)
    n_eval    = max(1, int(len(df) * eval_pct))
    df_eval   = df[:n_eval].sort_values("audio_file")
    df_train  = df[n_eval:].sort_values("audio_file")

    df_train.to_csv(str(train_csv), sep="|", index=False)
    df_eval.to_csv(str(eval_csv),   sep="|", index=False)

    print(f"  Train: {len(df_train)} samples")
    print(f"  Eval:  {len(df_eval)} samples")
    print(f"  Saved: {train_csv}")

    # Cleanup whisper from GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(train_csv), str(eval_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav",     required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lang",    default="en")
    args = p.parse_args()
    prepare(args.wav, args.out_dir, args.lang)
