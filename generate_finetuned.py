"""
Generate 5 sentences for all fine-tuned models.
Fixes the CUDA device ordinal error by using cuda:0 always
(CUDA_VISIBLE_DEVICES handles which physical GPU).
"""
import os, sys, json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

BASE       = Path("/scratch/s25089/voice_analysis")
MODELS_DIR = BASE / "finetuned_models"
CLONES_DIR = BASE / "clones" / "output_voices" / "xtts_finetune"
CLONES_DIR.mkdir(parents=True, exist_ok=True)

SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

# Always use cuda:0 — CUDA_VISIBLE_DEVICES controls which physical GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def find_best_checkpoint(model_dir: Path):
    ckpts = sorted(model_dir.glob("**/best_model_*.pth"),
                   key=lambda x: int(x.stem.split("_")[-1]))
    return ckpts[-1] if ckpts else None

def find_config(model_dir: Path):
    # Config is in the XTTS original files downloaded during training
    configs = list(model_dir.glob("**/config.json"))
    return str(configs[0]) if configs else None

def find_tokenizer(model_dir: Path):
    toks = list(model_dir.glob("**/vocab.json"))
    return str(toks[0]) if toks else None

def find_speaker_ref(model_dir: Path):
    # Use the longest chunk as speaker reference
    chunks = sorted(
        (model_dir / "dataset" / "wavs").glob("*.wav"),
        key=lambda x: x.stat().st_size, reverse=True
    )
    return str(chunks[0]) if chunks else None

def generate_for_person(person: str):
    model_dir = MODELS_DIR / person
    clone_dir = CLONES_DIR / person
    clone_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(clone_dir.glob("sent*.wav"))
    if len(existing) >= len(SENTENCES):
        print(f"  [skip] {person} — already generated")
        return True

    ckpt    = find_best_checkpoint(model_dir)
    config  = find_config(model_dir)
    tok     = find_tokenizer(model_dir)
    ref     = find_speaker_ref(model_dir)

    if not ckpt:
        print(f"  [skip] {person} — no checkpoint found")
        return False

    print(f"\n  {person}: checkpoint={ckpt.name}, device={DEVICE}")

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        cfg   = XttsConfig()
        cfg.load_json(config)
        model = Xtts.init_from_config(cfg)
        model.load_checkpoint(cfg,
            checkpoint_path = str(ckpt),
            vocab_path      = tok,
            use_deepspeed   = False,
        )
        model.to(DEVICE).eval()
        print(f"  Model loaded on {DEVICE}")

        gpt_lat, spk_emb = model.get_conditioning_latents(
            audio_path     = [ref],
            gpt_cond_len   = 30,
            max_ref_length = 60,
        )

        for key, text in SENTENCES.items():
            out = clone_dir / f"{key}.wav"
            if out.exists():
                print(f"  [skip] {key}")
                continue
            result = model.inference(
                text              = text,
                language          = "en",
                gpt_cond_latent   = gpt_lat,
                speaker_embedding = spk_emb,
                temperature       = 0.65,
                repetition_penalty= 10.0,
                top_k             = 50,
                top_p             = 0.85,
            )
            audio = np.array(result["wav"], dtype=np.float32)
            audio /= np.max(np.abs(audio)) + 1e-8
            sf.write(str(out), audio, 24000)
            print(f"  {key} OK")

        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"  FAILED {person}: {e}")
        return False

if __name__ == "__main__":
    # Get all people from models dir
    people = sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(people)} people: {people}")
    print(f"Device: {DEVICE}")

    ok, fail = 0, 0
    for person in people:
        success = generate_for_person(person)
        if success: ok += 1
        else: fail += 1

    print(f"\nDone: {ok} OK, {fail} failed")
    print(f"Clones at: {CLONES_DIR}")

if __name__ == "__main__" and len(sys.argv) > 1:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--person", required=True)
    p.add_argument("--gpu",    default="0")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    generate_for_person(args.person)
