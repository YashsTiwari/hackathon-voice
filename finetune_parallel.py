"""
finetune_parallel.py — Parallel XTTS v2 Fine-tuning on H200
─────────────────────────────────────────────────────────────
Auto-detects free GPUs, assigns one person per GPU,
runs all fine-tunes in parallel as subprocesses.

Usage:
    source /scratch/s25089/venvs/finetune/bin/activate
    python finetune_parallel.py

Output:
    /scratch/s25089/voice_analysis/finetuned_models/{person}/
    /scratch/s25089/voice_analysis/clones/xtts_finetune/{person}/sent1-5.wav
"""

import os, sys, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────
BASE        = Path("/scratch/s25089/voice_analysis")
REAL_DIR    = BASE / "real" / "input_voices"
MODELS_DIR  = BASE / "finetuned_models"
CLONES_DIR  = BASE / "clones" / "output_voices" / "xtts_finetune"
LOGS_DIR    = BASE / "finetune_logs"
WORKER      = Path(__file__).parent / "finetune_worker.py"
PYTHON      = "/scratch/s25089/venvs/finetune/bin/python"

for d in [MODELS_DIR, CLONES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sentences to generate after fine-tuning ───────────────────────
SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

# ── GPU selection — skip busy ones ────────────────────────────────
MIN_FREE_MB  = 20000   # need at least 20GB free
MAX_UTIL_PCT = 30      # skip if GPU util > 30%

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
        idx, mem_free, util = int(parts[0]), int(parts[1]), int(parts[2])
        if mem_free >= MIN_FREE_MB and util <= MAX_UTIL_PCT:
            free.append((idx, mem_free))

    # Sort by most free memory first
    free.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in free]

# ── Find voice files ───────────────────────────────────────────────
def find_voices():
    """Returns list of (person_name, wav_path) using converted WAVs."""
    voices = []
    seen   = set()
    for f in sorted(REAL_DIR.iterdir()):
        if "_converted" in f.stem and f.suffix == ".wav":
            # e.g. Abhishek_converted.wav → abhishek
            name = f.stem.replace("_converted", "").lower()
            name = name.replace("_vlsi", "").replace("_voice", "") \
                       .replace("_test", "").strip("_- ")
            if name not in seen:
                seen.add(name)
                voices.append((name, f))
    return voices

# ── Launch one fine-tune process ──────────────────────────────────
def launch(person, wav_path, gpu_id):
    model_out = MODELS_DIR / person
    clone_out = CLONES_DIR / person
    model_out.mkdir(parents=True, exist_ok=True)
    clone_out.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / f"{person}.log"

    cmd = [
        PYTHON, str(WORKER),
        "--person",   person,
        "--wav",      str(wav_path),
        "--model_out",str(model_out),
        "--clone_out",str(clone_out),
        "--gpu",      str(gpu_id),
        "--sentences",json.dumps(SENTENCES),
    ]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    return proc, log_path

# ── Monitor running processes ─────────────────────────────────────
def monitor(procs):
    """Wait for all processes, print status updates every 60s."""
    start   = time.time()
    running = dict(procs)  # {person: (proc, log_path)}

    while running:
        time.sleep(30)
        done = []
        for person, (proc, log) in running.items():
            ret = proc.poll()
            if ret is not None:
                status = "OK" if ret == 0 else f"FAILED (code {ret})"
                elapsed = (time.time() - start) / 60
                print(f"  [{elapsed:.0f}min] {person:15s} → {status}")
                print(f"           log: {log}")
                done.append(person)

        for p in done:
            del running[p]

        if running:
            elapsed = (time.time() - start) / 60
            print(f"  [{elapsed:.0f}min] Still running: {list(running.keys())}")

# ── Main ──────────────────────────────────────────────────────────
def main():
    voices   = find_voices()
    free_gpus = get_free_gpus()

    print(f"\n{'='*60}")
    print(f"  Parallel XTTS Fine-tuning")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"\n  People   : {len(voices)}")
    print(f"  Free GPUs: {free_gpus}")

    if not free_gpus:
        print("ERROR: No free GPUs found. Try again later.")
        sys.exit(1)

    if not voices:
        print(f"ERROR: No converted WAV files in {REAL_DIR}")
        sys.exit(1)

    # Skip already completed
    pending = []
    for person, wav in voices:
        clone_dir = CLONES_DIR / person
        done_files = list(clone_dir.glob("sent*.wav"))
        if len(done_files) >= len(SENTENCES):
            print(f"  [skip] {person} — already fine-tuned")
        else:
            pending.append((person, wav))

    if not pending:
        print("\n  All people already fine-tuned.")
        sys.exit(0)

    print(f"\n  Pending  : {[p for p,_ in pending]}")
    print(f"\n  Launching in batches of {len(free_gpus)} ...\n")

    # Process in batches matching available GPUs
    all_procs = {}
    for i in range(0, len(pending), len(free_gpus)):
        batch = pending[i : i + len(free_gpus)]
        print(f"  Batch {i//len(free_gpus)+1}: {[p for p,_ in batch]}")

        batch_procs = {}
        for j, (person, wav) in enumerate(batch):
            gpu = free_gpus[j % len(free_gpus)]
            proc, log = launch(person, wav, gpu)
            batch_procs[person] = (proc, log)
            print(f"    {person:15s} → GPU {gpu}  (log: {log.name})")

        all_procs.update(batch_procs)

        # Wait for this batch before starting next
        print(f"\n  Waiting for batch to complete...")
        monitor(batch_procs)
        print(f"  Batch done.\n")

    print(f"\n{'='*60}")
    print(f"  All fine-tuning complete.")
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Clones  : {CLONES_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
