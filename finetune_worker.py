"""
finetune_worker.py — Single person XTTS fine-tune worker (corrected)
Uses the official Coqui XTTS fine-tuning pipeline exactly as designed.

Steps:
  1. Install faster-whisper if needed
  2. Format audio using formatter.py (Whisper transcription + chunking)
  3. Fine-tune using gpt_train.py
  4. Generate 5 sentences using fine-tuned model
"""

import os, sys, json, argparse, warnings, gc
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths to official XTTS fine-tune utilities ────────────────────
import TTS
TTS_DIR      = Path(TTS.__file__).parent
FORMATTER_PY = TTS_DIR / "demos/xtts_ft_demo/utils/formatter.py"
GPT_TRAIN_PY = TTS_DIR / "demos/xtts_ft_demo/utils/gpt_train.py"

# Import directly from the TTS package
sys.path.insert(0, str(TTS_DIR / "demos/xtts_ft_demo/utils"))
from formatter import format_audio_list, list_audios
from gpt_train  import train_gpt

# ── Sentences ─────────────────────────────────────────────────────
DEFAULT_SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}


# ── Step 1: Format audio (Whisper transcription + chunking) ───────

def prepare_dataset(wav_path: str, out_dir: Path,
                    language: str = "en") -> tuple[str, str]:
    """
    Use the official formatter to:
    - Transcribe audio with Whisper large-v2
    - Split into sentence-level chunks
    - Create metadata_train.csv and metadata_eval.csv
    Returns (train_csv_path, eval_csv_path)
    """
    print(f"\n  [Step 1] Preparing dataset with Whisper transcription...")
    print(f"  Input: {wav_path}")
    print(f"  Output dir: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    train_csv = out_dir / "metadata_train.csv"
    eval_csv  = out_dir / "metadata_eval.csv"
    if train_csv.exists() and eval_csv.exists():
        print("  [skip] Dataset already prepared.")
        return str(train_csv), str(eval_csv)

    # Run formatter
    audio_files = [wav_path]
    train_meta, eval_meta, total_size = format_audio_list(
        audio_files    = audio_files,
        target_language= language,
        out_path       = str(out_dir),
        buffer         = 0.2,
        eval_percentage= 0.15,
        speaker_name   = "speaker",
        gradio_progress= None,
    )

    print(f"  Total audio processed: {total_size:.1f}s")
    print(f"  Train CSV: {train_meta}")
    print(f"  Eval CSV:  {eval_meta}")

    # Count samples
    import pandas as pd
    df_train = pd.read_csv(train_meta, sep="|")
    df_eval  = pd.read_csv(eval_meta,  sep="|")
    print(f"  Train samples: {len(df_train)}")
    print(f"  Eval samples:  {len(df_eval)}")

    return train_meta, eval_meta


# ── Step 2: Fine-tune GPT ──────────────────────────────────────────

def finetune(train_csv: str, eval_csv: str,
             model_out: Path, gpu_id: int,
             num_epochs: int = 6,
             batch_size: int = 4,
             grad_accum: int = 2) -> tuple:
    """
    Run XTTS GPT fine-tuning using official train_gpt function.
    Returns (config_path, checkpoint_path, tokenizer_path,
             trainer_out_path, speaker_ref)
    """
    print(f"\n  [Step 2] Fine-tuning XTTS v2 GPT...")
    print(f"  Epochs: {num_epochs}  Batch: {batch_size}  GradAccum: {grad_accum}")
    print(f"  GPU: cuda:{gpu_id}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = train_gpt(
        language         = "en",
        num_epochs       = num_epochs,
        batch_size       = batch_size,
        grad_acumm       = grad_accum,
        train_csv        = train_csv,
        eval_csv         = eval_csv,
        output_path      = str(model_out),
        max_audio_length = 255995,  # ~11.6 seconds
    )

    config_path, checkpoint_path, tokenizer_path, trainer_out, speaker_ref = result
    print(f"\n  Fine-tune complete.")
    print(f"  Checkpoint: {trainer_out}")
    print(f"  Speaker ref: {speaker_ref}")

    return config_path, checkpoint_path, tokenizer_path, trainer_out, speaker_ref


# ── Step 3: Generate clones using fine-tuned model ────────────────

def generate_clones(config_path: str,
                    checkpoint_dir: str,
                    tokenizer_path: str,
                    speaker_ref: str,
                    clone_out: Path,
                    sentences: dict,
                    gpu_id: int):
    """
    Generate sentences using the fine-tuned XTTS model.
    """
    print(f"\n  [Step 3] Generating {len(sentences)} sentences...")
    clone_out.mkdir(parents=True, exist_ok=True)

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Find the best checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints    = sorted(checkpoint_dir.glob("**/*.pth"))
    if not checkpoints:
        # Try trainer output path
        checkpoints = sorted(Path(checkpoint_dir).parent.glob("**/*.pth"))

    if not checkpoints:
        print("  WARNING: No checkpoint found, falling back to base model")
        _generate_with_base_model(speaker_ref, clone_out, sentences, gpu_id)
        return

    best_ckpt = checkpoints[-1]
    print(f"  Loading checkpoint: {best_ckpt}")

    # Load fine-tuned model
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path = str(best_ckpt),
        vocab_path      = tokenizer_path,
        use_deepspeed   = False,
    )
    model.to(device)
    model.eval()

    # Get speaker conditioning
    print(f"  Computing speaker latents from: {speaker_ref}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path    = [speaker_ref],
        gpt_cond_len  = 30,
        max_ref_length= 60,
    )

    # Generate each sentence
    for sent_key, text in sentences.items():
        out_path = clone_out / f"{sent_key}.wav"
        if out_path.exists():
            print(f"  [skip] {sent_key}.wav")
            continue

        print(f"  Generating {sent_key}: {text[:50]}...")
        try:
            out = model.inference(
                text              = text,
                language          = "en",
                gpt_cond_latent   = gpt_cond_latent,
                speaker_embedding = speaker_embedding,
                temperature       = 0.65,
                repetition_penalty= 10.0,
                top_k             = 50,
                top_p             = 0.85,
            )
            audio = np.array(out["wav"], dtype=np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            sf.write(str(out_path), audio, 24000)
            print(f"  {sent_key} OK → {out_path.name}")
        except Exception as e:
            print(f"  {sent_key} FAILED: {e}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _generate_with_base_model(speaker_ref, clone_out, sentences, gpu_id):
    """Fallback: use base XTTS with speaker reference."""
    from TTS.api import TTS
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    tts    = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    for sent_key, text in sentences.items():
        out_path = clone_out / f"{sent_key}.wav"
        if out_path.exists():
            continue
        audio = tts.tts(text=text, speaker_wav=speaker_ref, language="en")
        audio = np.array(audio, dtype=np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        sf.write(str(out_path), audio, 24000)
        print(f"  {sent_key} OK (base model)")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",    required=True)
    parser.add_argument("--wav",       required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--clone_out", required=True)
    parser.add_argument("--gpu",       default="0")
    parser.add_argument("--sentences", default=json.dumps(DEFAULT_SENTENCES))
    parser.add_argument("--epochs",    default="6",  type=int)
    parser.add_argument("--batch",     default="4",  type=int)
    parser.add_argument("--grad_accum",default="2",  type=int)
    args = parser.parse_args()

    gpu_id    = int(args.gpu)
    sentences = json.loads(args.sentences)
    model_out = Path(args.model_out)
    clone_out = Path(args.clone_out)
    dataset_dir = model_out / "dataset"

    print(f"\n{'='*55}")
    print(f"  XTTS Fine-tuning: {args.person}")
    print(f"  GPU:    cuda:{gpu_id}")
    print(f"  Audio:  {args.wav}")
    print(f"  Output: {model_out}")
    print(f"{'='*55}")

    import time
    t_start = time.time()

    # Step 1 — prepare dataset
    train_csv, eval_csv = prepare_dataset(args.wav, dataset_dir)

    # Step 2 — fine-tune
    config_path, ckpt_path, tok_path, trainer_out, speaker_ref = finetune(
        train_csv  = train_csv,
        eval_csv   = eval_csv,
        model_out  = model_out,
        gpu_id     = gpu_id,
        num_epochs = args.epochs,
        batch_size = args.batch,
        grad_accum = args.grad_accum,
    )

    # Step 3 — generate
    generate_clones(
        config_path     = config_path,
        checkpoint_dir  = trainer_out,
        tokenizer_path  = tok_path,
        speaker_ref     = speaker_ref,
        clone_out       = clone_out,
        sentences       = sentences,
        gpu_id          = gpu_id,
    )

    total = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print(f"  {args.person} complete in {total:.1f} min")
    print(f"  Clones: {clone_out}")
    print(f"{'='*55}")