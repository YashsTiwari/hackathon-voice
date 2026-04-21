import subprocess, sys
from pathlib import Path

VOICES = [
    ("abhishek",     "real/input_voices/Abhishek_converted.wav"),
    ("abhishek2",    "real/input_voices/Abhishek_VLSI_converted.wav"),
    ("bhavya",       "real/input_voices/Bhavya_converted.wav"),
    ("harshvardhan", "real/input_voices/Harshvardhan_converted.wav"),
    ("saksham",      "real/input_voices/Saksham_converted.wav"),
    ("sezal",        "real/input_voices/Sezal_converted.wav"),
    ("shagun",       "real/input_voices/shagun_converted.wav"),
    ("shalini",      "real/input_voices/Shalini_converted.wav"),
    ("tarun",        "real/input_voices/Tarun_converted.wav"),
    ("yashs",        "real/input_voices/Yashs_converted.wav"),
]

BASE = Path("finetuned_models")
PY   = sys.executable

for person, wav in VOICES:
    train_csv = BASE / person / "dataset" / "metadata_train.csv"
    if train_csv.exists():
        print(f"[skip] {person}")
        continue
    print(f"\n>>> Transcribing: {person}")
    r = subprocess.run([
        PY, "prepare_dataset.py",
        "--wav",     wav,
        "--out_dir", str(BASE / person / "dataset"),
    ], capture_output=False)
    if r.returncode != 0:
        print(f"  FAILED: {person}")
    else:
        print(f"  Done: {person}")

print("\nAll transcriptions complete.")
