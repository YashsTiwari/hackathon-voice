"""
Generate fine-tuned clones in parallel — one person per GPU.
"""
import os, sys, json, subprocess
from pathlib import Path

BASE       = Path("/scratch/s25089/voice_analysis")
MODELS_DIR = BASE / "finetuned_models"
CLONES_DIR = BASE / "clones" / "output_voices" / "xtts_finetune"
LOGS_DIR   = BASE / "finetune_logs"
PYTHON     = sys.executable

SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

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
        if mem >= 15000 and util <= 40:
            free.append(idx)
    return free

def is_done(person):
    clone_dir = CLONES_DIR / person
    return clone_dir.exists() and len(list(clone_dir.glob("sent*.wav"))) >= 5

def launch(person, gpu_id):
    log = LOGS_DIR / f"gen_{person}.log"
    cmd = [
        PYTHON, "generate_finetuned.py",
        "--person", person,
        "--gpu",    "0",  # always 0 inside subprocess
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    f   = open(log, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), log

if __name__ == "__main__":
    import time

    people  = sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir()])
    pending = [p for p in people if not is_done(p)]

    print(f"People: {len(people)}  Pending: {len(pending)}")
    if not pending:
        print("All done already.")
        sys.exit(0)

    free = get_free_gpus()
    print(f"Free GPUs: {free}")

    procs = {}
    gpu_i = 0

    for person in pending:
        gpu = free[gpu_i % len(free)]
        gpu_i += 1
        print(f"  {person:15s} → GPU {gpu}")
        proc, log = launch(person, gpu)
        procs[person] = (proc, log, time.time())

    print(f"\nWaiting for {len(procs)} generation jobs...")
    while procs:
        time.sleep(15)
        done = []
        for person, (proc, log, t0) in procs.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = (time.time() - t0) / 60
                status  = "OK" if ret == 0 else f"FAILED({ret})"
                clones  = len(list((CLONES_DIR/person).glob("sent*.wav"))) \
                          if (CLONES_DIR/person).exists() else 0
                print(f"  [{elapsed:.0f}min] {person:15s} → {status} | {clones} clones")
                done.append(person)
        for p in done:
            del procs[p]
        if procs:
            print(f"  Still running: {list(procs.keys())}")

    print("\nAll generation complete.")
    total = sum(len(list((CLONES_DIR/p).glob("sent*.wav")))
                for p in people if (CLONES_DIR/p).exists())
    print(f"Total clones generated: {total}")
