import argparse, warnings, time
import scipy.io.wavfile
from pathlib import Path
warnings.filterwarnings("ignore")

SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref",     required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--person",  required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir) / args.person
    out_dir.mkdir(parents=True, exist_ok=True)

    from pocket_tts import TTSModel
    print(f"Loading PocketTTS...")
    model = TTSModel.load_model()
    print(f"Loading voice: {args.ref}")
    voice_state = model.get_state_for_audio_prompt(args.ref)

    for sent_key, text in SENTENCES.items():
        out_path = out_dir / f"{sent_key}.wav"
        if out_path.exists():
            print(f"  [skip] {sent_key}.wav")
            continue
        print(f"  {sent_key} ...", end=" ", flush=True)
        t0 = time.time()
        audio = model.generate_audio(voice_state, text)
        scipy.io.wavfile.write(str(out_path), model.sample_rate, audio.numpy())
        print(f"OK ({time.time()-t0:.1f}s)")

    print(f"Done → {out_dir}")
