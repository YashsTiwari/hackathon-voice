"""
finetune_all.py — Parallel XTTS fine-tuning for all people
Uses prepare_dataset.py transcripts (already done)
Auto-assigns people to free GPUs, runs in parallel batches
"""

import os, sys, json, time, subprocess
from pathlib import Path
from datetime import datetime

BASE       = Path("/scratch/s25089/voice_analysis")
MODELS_DIR = BASE / "finetuned_models"
CLONES_DIR = BASE / "clones" / "output_voices" / "xtts_finetune"
LOGS_DIR   = BASE / "finetune_logs"
PYTHON     = sys.executable

LOGS_DIR.mkdir(parents=True, exist_ok=True)
CLONES_DIR.mkdir(parents=True, exist_ok=True)

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

SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

MIN_FREE_MB  = 20000
MAX_UTIL_PCT = 30

def get_free_gpus():
    r = subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], capture_output=True, text=True)
    free = []
    for line in r.stdout.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        idx, mem, util = int(parts[0]), int(parts[1]), int(parts[2])
        if mem >= MIN_FREE_MB and util <= MAX_UTIL_PCT:
            free.append((idx, mem))
    free.sort(key=lambda x: x[1], reverse=True)
    print(f"  Free GPUs: {[f'GPU{i}({m//1024}GB)' for i,m in free]}")
    return [i for i, _ in free]

def is_done(person):
    clone_dir = CLONES_DIR / person
    done = list(clone_dir.glob("sent*.wav")) if clone_dir.exists() else []
    return len(done) >= len(SENTENCES)

def launch_one(person, wav, gpu_id):
    model_out = MODELS_DIR / person
    clone_out = CLONES_DIR / person
    log_path  = LOGS_DIR / f"{person}.log"

    cmd = [
        PYTHON, "finetune_worker_v2.py",
        "--person",    person,
        "--wav",       wav,
        "--model_out", str(model_out),
        "--clone_out", str(clone_out),
        "--gpu",       str(gpu_id),
        "--sentences", json.dumps(SENTENCES),
        "--epochs",    "6",
        "--batch",     "8",
        "--grad_accum","1",
    ]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log_path

def run_batch(batch, gpu_ids):
    procs = {}
    for i, (person, wav) in enumerate(batch):
        gpu = gpu_ids[i % len(gpu_ids)]
        print(f"  {person:15s} → GPU {gpu}")
        proc, log = launch_one(person, wav, gpu)
        procs[person] = (proc, log, time.time())

    print(f"\n  Waiting for batch: {list(procs.keys())}")
    while procs:
        time.sleep(20)
        done = []
        for person, (proc, log, t0) in procs.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = (time.time() - t0) / 60
                status  = "OK" if ret == 0 else f"FAILED(code={ret})"
                print(f"  [{elapsed:.0f}min] {person:15s} → {status} | log: {log}")
                done.append(person)
        for p in done:
            del procs[p]
        if procs:
            elapsed = (time.time() - list(procs.values())[0][2]) / 60
            print(f"  [{elapsed:.0f}min] Still running: {list(procs.keys())}")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Parallel XTTS Fine-tuning — {datetime.now().strftime('%H:%M')}")
    print(f"{'='*60}\n")

    # Find pending people
    pending = []
    for person, wav in VOICES:
        if is_done(person):
            print(f"  [done] {person}")
        else:
            pending.append((person, wav))

    if not pending:
        print("  All people already fine-tuned.")
        sys.exit(0)

    print(f"\n  Pending: {[p for p,_ in pending]}")

    free_gpus = get_free_gpus()
    if not free_gpus:
        print("  ERROR: No free GPUs")
        sys.exit(1)

    # Run in batches
    batch_size = len(free_gpus)
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        print(f"\n  Batch {i//batch_size+1}: {[p for p,_ in batch]}")
        run_batch(batch, free_gpus)
        print(f"  Batch {i//batch_size+1} complete.\n")

    print(f"\n{'='*60}")
    print(f"  All fine-tuning complete.")
    print(f"  Clones: {CLONES_DIR}")
    print(f"  Check logs: {LOGS_DIR}")
    print(f"{'='*60}")
