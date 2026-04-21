"""
finetune_worker_v2.py — Single person XTTS fine-tune
Uses prepare_dataset.py (openai-whisper, no faster-whisper)
Skips transcription if already done
"""

import os, sys, json, argparse, warnings, gc, time
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

import TTS
TTS_DIR = Path(TTS.__file__).parent
sys.path.insert(0, str(TTS_DIR / "demos/xtts_ft_demo/utils"))
from gpt_train import train_gpt

SENTENCES_DEFAULT = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

def prepare_dataset(wav_path, model_out, gpu_id):
    """Run prepare_dataset.py if not already done."""
    dataset_dir = model_out / "dataset"
    train_csv   = dataset_dir / "metadata_train.csv"
    eval_csv    = dataset_dir / "metadata_eval.csv"

    if train_csv.exists() and eval_csv.exists():
        import pandas as pd
        df = pd.read_csv(train_csv, sep="|")
        print(f"  [skip] Dataset ready — {len(df)} train samples")
        return str(train_csv), str(eval_csv)

    print(f"  Preparing dataset...")
    r = subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "prepare_dataset.py"),
        "--wav",     wav_path,
        "--out_dir", str(dataset_dir),
    ], capture_output=False,
       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})

    if r.returncode != 0:
        raise RuntimeError("prepare_dataset.py failed")

    return str(train_csv), str(eval_csv)

def finetune(train_csv, eval_csv, model_out, gpu_id, epochs, batch, grad_accum):
    print(f"\n  Fine-tuning on GPU {gpu_id}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = train_gpt(
        language         = "en",
        num_epochs       = epochs,
        batch_size       = batch,
        grad_acumm       = grad_accum,
        train_csv        = train_csv,
        eval_csv         = eval_csv,
        output_path      = str(model_out),
        max_audio_length = 255995,
    )
    return result

def generate(config_path, ckpt_dir, tok_path, speaker_ref,
             clone_out, sentences, gpu_id):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    clone_out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Find checkpoint
    ckpts = sorted(Path(ckpt_dir).glob("**/*.pth"))
    if not ckpts:
        print("  No checkpoint — using base model")
        _generate_base(speaker_ref, clone_out, sentences, gpu_id)
        return

    print(f"  Loading checkpoint: {ckpts[-1].name}")
    config = XttsConfig()
    config.load_json(config_path)
    model  = Xtts.init_from_config(config)
    model.load_checkpoint(config,
        checkpoint_path=str(ckpts[-1]),
        vocab_path=tok_path,
        use_deepspeed=False)
    model.to(device).eval()

    gpt_lat, spk_emb = model.get_conditioning_latents(
        audio_path=[speaker_ref], gpt_cond_len=30, max_ref_length=60)

    for key, text in sentences.items():
        out = clone_out / f"{key}.wav"
        if out.exists():
            print(f"  [skip] {key}.wav")
            continue
        try:
            result = model.inference(
                text=text, language="en",
                gpt_cond_latent=gpt_lat,
                speaker_embedding=spk_emb,
                temperature=0.65, repetition_penalty=10.0,
                top_k=50, top_p=0.85)
            audio = np.array(result["wav"], dtype=np.float32)
            audio /= np.max(np.abs(audio)) + 1e-8
            sf.write(str(out), audio, 24000)
            print(f"  {key} OK")
        except Exception as e:
            print(f"  {key} FAILED: {e}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

def _generate_base(speaker_ref, clone_out, sentences, gpu_id):
    from TTS.api import TTS
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    tts    = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    for key, text in sentences.items():
        out = clone_out / f"{key}.wav"
        if out.exists(): continue
        audio = np.array(tts.tts(text=text, speaker_wav=speaker_ref,
                                  language="en"), dtype=np.float32)
        audio /= np.max(np.abs(audio)) + 1e-8
        sf.write(str(out), audio, 24000)
        print(f"  {key} OK (base)")

if __name__ == "__main__":
    import subprocess
    p = argparse.ArgumentParser()
    p.add_argument("--person");    p.add_argument("--wav")
    p.add_argument("--model_out"); p.add_argument("--clone_out")
    p.add_argument("--gpu",        default="0")
    p.add_argument("--sentences",  default=json.dumps(SENTENCES_DEFAULT))
    p.add_argument("--epochs",     default=6,  type=int)
    p.add_argument("--batch",      default=8,  type=int)
    p.add_argument("--grad_accum", default=1,  type=int)
    args = p.parse_args()

    gpu       = int(args.gpu)
    sentences = json.loads(args.sentences)
    model_out = Path(args.model_out)
    clone_out = Path(args.clone_out)

    print(f"\n{'='*50}")
    print(f"  {args.person} | GPU {gpu}")
    print(f"{'='*50}")

    t_start = time.time()

    # Step 1 — dataset (likely already done)
    train_csv, eval_csv = prepare_dataset(args.wav, model_out, gpu)

    # Step 2 — fine-tune
    config_path, ckpt_path, tok_path, trainer_out, speaker_ref = finetune(
        train_csv, eval_csv, model_out, gpu,
        args.epochs, args.batch, args.grad_accum)

    # Step 3 — generate
    generate(config_path, trainer_out, tok_path, speaker_ref,
             clone_out, sentences, gpu)

    total = (time.time() - t_start) / 60
    print(f"\n  Done: {args.person} in {total:.1f}min")
