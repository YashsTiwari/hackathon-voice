import sys
sys.path.insert(0, "/scratch/s25089/venvs/finetune/lib/python3.10/site-packages/TTS/demos/xtts_ft_demo/utils")
from formatter import format_audio_list
from pathlib import Path
import pandas as pd

VOICES = [
    ("abhishek",    "real/input_voices/Abhishek_converted.wav"),
    ("abhishek2",   "real/input_voices/Abhishek_VLSI_converted.wav"),
    ("bhavya",      "real/input_voices/Bhavya_converted.wav"),
    ("harshvardhan","real/input_voices/Harshvardhan_converted.wav"),
    ("saksham",     "real/input_voices/Saksham_converted.wav"),
    ("sezal",       "real/input_voices/Sezal_converted.wav"),
    ("shagun",      "real/input_voices/shagun_converted.wav"),
    ("shalini",     "real/input_voices/Shalini_converted.wav"),
    ("tarun",       "real/input_voices/Tarun_converted.wav"),
    ("yashs",       "real/input_voices/Yashs_converted.wav"),
]

BASE = Path("/scratch/s25089/voice_analysis/finetuned_models")

for person, wav in VOICES:
    dataset_dir = BASE / person / "dataset"
    train_csv   = dataset_dir / "metadata_train.csv"
    eval_csv    = dataset_dir / "metadata_eval.csv"

    if train_csv.exists() and eval_csv.exists():
        df = pd.read_csv(train_csv, sep="|")
        print(f"[skip] {person} — already transcribed ({len(df)} samples)")
        continue

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Transcribing: {person} ({wav})")

    try:
        train_meta, eval_meta, total = format_audio_list(
            audio_files     = [wav],
            target_language = "en",
            out_path        = str(dataset_dir),
            buffer          = 0.2,
            eval_percentage = 0.15,
            speaker_name    = "speaker",
        )
        df = pd.read_csv(train_meta, sep="|")
        print(f"  Done: {total:.0f}s audio, {len(df)} train samples")
    except Exception as e:
        print(f"  FAILED: {e}")

print("\nAll transcriptions complete.")
