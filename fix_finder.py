# Run this to patch find_all_files in deep_analysis.py
import re

with open("deep_analysis.py", "r") as f:
    content = f.read()

new_finder = '''
def find_all_files() -> list[dict]:
    """Find all real and fake audio files."""
    entries = []

    # Build real voice lookup: person_name → filepath
    real_lookup = {}
    for wav in REAL_DIR.glob("*_converted.wav"):
        name = wav.stem.lower()
        for suffix in ["_converted","_vlsi","_voice","_test","_audio"]:
            name = name.replace(suffix, "")
        name = name.strip("_- ")
        real_lookup[name] = str(wav)

    print(f"  Real voices found: {list(real_lookup.keys())}")

    # Fake clones
    for system in SYSTEMS:
        sys_dir = CLONE_DIR / system
        if not sys_dir.exists():
            print(f"  [skip] {system} — folder not found")
            continue
        for person_dir in sorted(sys_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            wavs = list(person_dir.glob("sent*.wav"))
            if not wavs:
                continue
            for wav in sorted(wavs):
                sent = wav.stem
                entries.append({
                    "person":   person,
                    "system":   system,
                    "sentence": sent,
                    "label":    "fake",
                    "filepath": str(wav),
                })
            # Add matching real voice once per person
            if person in real_lookup:
                real_already = any(
                    e["person"] == person and e["system"] == "real"
                    for e in entries
                )
                if not real_already:
                    entries.append({
                        "person":   person,
                        "system":   "real",
                        "sentence": "full",
                        "label":    "real",
                        "filepath": real_lookup[person],
                    })
            else:
                print(f"  [warn] No real voice found for: {person}")

    print(f"  Total entries: {len(entries)}")
    real_count = sum(1 for e in entries if e["label"] == "real")
    fake_count = sum(1 for e in entries if e["label"] == "fake")
    print(f"  Real: {real_count}  Fake: {fake_count}")
    return entries
'''

# Replace old find_all_files
old = content[content.find("def find_all_files"):
              content.find("\ndef main")]
content = content.replace(old, new_finder + "\n\n")

with open("deep_analysis.py", "w") as f:
    f.write(content)

print("Patched successfully.")
