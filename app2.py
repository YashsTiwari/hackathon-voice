import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import find_peaks

# --- Config & Theme ---
st.set_page_config(page_title="Voice Forensic Lab", layout="wide")

T = {
    "red": "#FF3B3B",
    "blue": "#4FC3F7",
    "bg": "#0B0C10",
    "card": "#141720",
    "text": "#E8EAF6"
}

# --- Backend ---
@st.cache_resource
def get_encoder():
    return VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

def process_audio(path):
    y, sr = librosa.load(path, sr=16000)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    rms = librosa.feature.rms(y=y)[0]
    shim = np.std(rms) / (np.mean(rms) + 1e-10)
    return y, sr, spec, shim

# --- 34 Features ---
def extract_34_features(y, sr):
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))

    feats.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats.append(np.mean(librosa.feature.spectral_flatness(y=y)))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.append(np.mean(zcr))

    rms = librosa.feature.rms(y=y)
    feats.append(np.mean(rms))
    feats.append(np.std(rms))

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[mags > np.median(mags)]
    feats.append(np.mean(pitch_vals) if len(pitch_vals) else 0)
    feats.append(np.std(pitch_vals) if len(pitch_vals) else 0)

    while len(feats) < 34:
        feats.append(0)

    return np.array(feats[:34])

def get_local_files(directory):
    p = Path(directory)
    if not p.exists(): return []
    return [f.name for f in p.glob("*.wav")]

# UI
st.title("🎙️ Voice Clone Forensic Dashboard")

col_main1, col_btn, col_main2 = st.columns([4, 1, 4])

with col_main1:
    st.markdown("### Real Audio Input")
    c1, c2 = st.columns(2)
    with c1: real_upload = st.file_uploader("Upload New", type=['wav'], key="u1")
    with c2:
        real_local = get_local_files("input_voices")
        real_choice = st.selectbox("Select Server File", ["None"] + real_local, key="s1")

    real_final = real_upload if real_upload else (f"input_voices/{real_choice}" if real_choice!="None" else None)
    if real_final: st.audio(real_final)

with col_main2:
    st.markdown("### Generated Audio Input")
    c3, c4 = st.columns(2)
    with c3: gen_upload = st.file_uploader("Upload New", type=['wav'], key="u2")
    with c4:
        gen_local = get_local_files("output_voices/xtts_finetune/shalini")
        gen_choice = st.selectbox("Select Server File", ["None"] + gen_local, key="s2")

    gen_final = gen_upload if gen_upload else (f"output_voices/xtts_finetune/shalini/{gen_choice}" if gen_choice!="None" else None)
    if gen_final: st.audio(gen_final)

with col_btn:
    st.write("")
    analyze_clicked = st.button("ANALYZE 🔍", use_container_width=True)

# ANALYSIS
if analyze_clicked and real_final and gen_final:

    path_r = "temp_real.wav" if hasattr(real_final, 'read') else real_final
    path_g = "temp_gen.wav" if hasattr(gen_final, 'read') else gen_final

    if hasattr(real_final, 'read'):
        with open(path_r, "wb") as f: f.write(real_final.getbuffer())
    if hasattr(gen_final, 'read'):
        with open(path_g, "wb") as f: f.write(gen_final.getbuffer())

    y_r, sr, spec_r, _ = process_audio(path_r)
    y_g, _, spec_g, _ = process_audio(path_g)

    encoder = get_encoder()
    emb_r = encoder.embed_utterance(preprocess_wav(path_r))
    emb_g = encoder.embed_utterance(preprocess_wav(path_g))
    cosine_sim = np.dot(emb_r, emb_g) / (np.linalg.norm(emb_r) * np.linalg.norm(emb_g))

    feat_r = extract_34_features(y_r, sr)
    feat_g = extract_34_features(y_g, sr)
    feature_dist = np.linalg.norm(feat_r - feat_g)

    st.divider()

    verdict = (
        "HIGH MATCH" if cosine_sim > 0.8 and feature_dist < 5 else
        "PROBABLE CLONE" if cosine_sim > 0.6 else
        "DISTINCT VOICES"
    )

    v_color = "green" if verdict == "HIGH MATCH" else "orange" if verdict == "PROBABLE CLONE" else "red"

    st.subheader("Final Similarity Report")
    st.markdown(f"### Verdict: :{v_color}[{verdict}]")

    st.info(f"""
Cosine Similarity: **{cosine_sim*100:.2f}%**  
Feature Distance (34 params): **{feature_dist:.2f}**
""")

    # Spectrograms
    c_spec1, c_metric, c_spec2 = st.columns([2,1,2])

    with c_spec1:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        librosa.display.specshow(spec_r, sr=sr, x_axis='time', ax=ax1, cmap='magma')
        st.pyplot(fig1)

    with c_metric:
        st.metric("SSIM / Cosine", f"{cosine_sim:.4f}")
        st.progress(float(np.clip(cosine_sim, 0, 1)))

    with c_spec2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        librosa.display.specshow(spec_g, sr=sr, x_axis='time', ax=ax2, cmap='magma')
        st.pyplot(fig2)

    # Middle
    m1, m2 = st.columns([1.5,2.5])

    with m1:
        st.subheader("Forensic Parameter Impact (34 Tiers)")
        param_diffs = np.abs(feat_r - feat_g)

        param_names = (
            [f"MFCC_Mean_{i}" for i in range(13)] +
            [f"MFCC_Std_{i}" for i in range(13)] +
            ["Centroid","Bandwidth","Flatness","ZCR","RMS_Mean","RMS_Std","Pitch_Mean","Pitch_Std"]
        )[:34]

        fig_bar = go.Figure(go.Bar(x=param_diffs, y=param_names, orientation='h', marker_color=T['blue']))
        fig_bar.update_layout(height=800, template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    with m2:
        st.subheader("Signal Overlap")
        fig_wave = go.Figure()
        step = max(1, len(y_r)//2000)
        fig_wave.add_trace(go.Scatter(y=y_r[::step], name="Real", line_color=T['red']))
        fig_wave.add_trace(go.Scatter(y=y_g[::step], name="Generated", line_color=T['blue']))
        fig_wave.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_wave, use_container_width=True)

        st.divider()
        st.subheader("Deep Signal Analysis")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.line_chart(librosa.onset.onset_strength(y=y_g, sr=sr))
        with col_g2:
            st.line_chart(librosa.feature.zero_crossing_rate(y_g)[0])

        # ✅ FINAL ADDITION (ALL EXTRA FIGURES)
        # ✅ FINAL ADDITION (FORENSIC-STYLE VISUALS)
        st.divider()
        st.subheader("Extended Forensic Signals")

        rms = librosa.feature.rms(y=y_g)[0]

        # ─────────────────────────────
        # 1. RMS ENERGY
        # ─────────────────────────────
        st.caption("RMS Energy (Voice Activity)")
        st.line_chart(rms)

        # ─────────────────────────────
        # 2. SNR WITH REGION HIGHLIGHT
        # ─────────────────────────────
        st.caption("SNR Behaviour (Voiced vs Silence)")

        thresh = np.mean(rms) * 0.3
        voiced = rms > thresh

        fig_snr, ax = plt.subplots()

        # 🔵 voiced / 🔴 silence regions
        for i in range(len(rms)-1):
            if voiced[i]:
                ax.axvspan(i, i+1, alpha=0.08, color='blue')
            else:
                ax.axvspan(i, i+1, alpha=0.08, color='red')

        ax.plot(rms, color="white", label="RMS")
        ax.axhline(thresh, linestyle="--", label="Threshold")

        ax.set_title("SNR Highlight (Silence Detection)")
        ax.legend()

        st.pyplot(fig_snr)

        # ─────────────────────────────
        # 3. SHIMMER WITH FAKE ZONE
        # ─────────────────────────────
        st.caption("Shimmer (Amplitude Stability)")

        shimmer_curve = np.array([
            np.std(rms[max(0,i-5):i+5])/(np.mean(rms[max(0,i-5):i+5])+1e-10)
            for i in range(len(rms))
        ])

        THRESH_SHIM = 0.02

        fig_shim, ax = plt.subplots()

        ax.plot(shimmer_curve, color="purple", label="Shimmer")

        # 🔴 fake region
        ax.fill_between(
            range(len(shimmer_curve)),
            shimmer_curve,
            THRESH_SHIM,
            where=(shimmer_curve < THRESH_SHIM),
            alpha=0.3
        )

        ax.axhline(THRESH_SHIM, linestyle="--", label="Threshold")

        ax.set_title("Shimmer Fake Zone Detection")
        ax.legend()

        st.pyplot(fig_shim)

        # ─────────────────────────────
        # 4. AMPLITUDE ENVELOPE
        # ─────────────────────────────
        st.caption("Amplitude Envelope")
        st.line_chart(np.abs(y_g))

        # ─────────────────────────────
        # 5. DCT WITH PEAK HIGHLIGHT
        # ─────────────────────────────
        st.caption("DCT Spectrum (Artifact Detection)")

        chunk = y_g[:min(len(y_g), sr*4)]
        dct = np.abs(np.fft.rfft(chunk, n=8192))

        peaks, _ = find_peaks(dct, height=np.percentile(dct, 90), distance=10)

        fig_dct, ax = plt.subplots()

        ax.plot(dct[:2000], label="Spectrum")
        ax.scatter(peaks, dct[peaks], color="red", label="Peaks")

        ax.set_title("DCT Peaks (Fake Artifacts)")
        ax.legend()

        st.pyplot(fig_dct)

        # ─────────────────────────────
        # 6. FORENSIC METRICS + TRIGGERS
        # ─────────────────────────────
        st.caption("Forensic Verdict Signals")

        snr_ratio = np.mean(rms[rms>thresh])/(np.mean(rms[rms<=thresh])+1e-10)
        shim_val = np.mean(shimmer_curve)
        dct_score = len(peaks)

        c1, c2, c3 = st.columns(3)

        c1.metric("SNR", f"{snr_ratio:.2f}", "⚠ TRIGGER" if snr_ratio > 3 else "OK")
        c2.metric("Shimmer", f"{shim_val:.4f}", "⚠ TRIGGER" if shim_val < 0.02 else "OK")
        c3.metric("DCT Peaks", f"{dct_score}", "⚠ TRIGGER" if dct_score > 50 else "OK")

        # ─────────────────────────────
        # 7. EXPLAINABILITY LAYER
        # ─────────────────────────────
        st.markdown("### 🔬 Forensic Interpretation")

        if snr_ratio > 3:
            st.error("SNR Triggered → Silence is unnaturally clean (TTS artifact)")

        if shim_val < 0.02:
            st.error("Shimmer Triggered → Voice lacks natural amplitude variation")

        if dct_score > 50:
            st.error("DCT Triggered → Periodic spectral peaks detected (upsampling artifact)")

        if not (snr_ratio > 3 or shim_val < 0.02 or dct_score > 50):
            st.success("No strong synthetic artifacts detected")




elif analyze_clicked:
    st.warning("Please ensure both files are selected.")