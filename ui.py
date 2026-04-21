import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.signal import find_peaks
from detector import VoiceDetector

# --- Config & Theme ---
st.set_page_config(
    page_title="Voice Forensic Lab",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Clean professional palette
T = {
    "red": "#E05252",
    "blue": "#5B8DEF",
    "green": "#4CAF82",
    "orange": "#E0944B",
    "purple": "#9B7FD4",
    "bg": "#111318",
    "card": "#191C24",
    "card_light": "#23272F",
    "text": "#D5D9E2",
    "text_muted": "#7C8190"
}

# Minimal CSS overrides
st.markdown(f"""
<style>
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    h1, h2, h3 {{
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    .stMetric {{
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Backend ---
@st.cache_resource
def get_encoder():
    return VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_voice_detector():
    return VoiceDetector(verbose=False)

def process_audio(path):
    y, sr = librosa.load(path, sr=16000)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    rms = librosa.feature.rms(y=y)[0]
    shim = np.std(rms) / (np.mean(rms) + 1e-10)
    return y, sr, spec, shim

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
st.title("Voice Clone Forensic Dashboard")
st.caption("Advanced voice authentication & deepfake detection")

# Sidebar Settings
with st.sidebar:
    st.markdown("## Settings")
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    st.markdown(f"**Device:** {device_info}")
    st.markdown("**Status:** Ready")
    
    st.divider()
    st.markdown("### Analysis Options")
    show_detailed = st.checkbox("Show Detailed Analysis", value=True)
    show_forensics = st.checkbox("Show Forensic Signals", value=True)
    show_metrics = st.checkbox("Show Metrics Dashboard", value=True)

# Main Tabs
tab1, tab3 = st.tabs(["Single Audio Analysis", "Comparison"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Analyze Single Audio File")
    st.caption("Upload or select an audio file for detailed forensic analysis")
    
    col_single1, col_single2 = st.columns(2)
    
    with col_single1:
        single_upload = st.file_uploader("Upload Audio", type=['wav'], key="single_upload")
    
    with col_single2:
        single_local = get_local_files("input_voices")
        single_choice = st.selectbox("Select Server File", ["None"] + single_local, key="single_select")
    
    single_final = single_upload if single_upload else (f"input_voices/{single_choice}" if single_choice!="None" else None)
    
    if single_final:
        st.audio(single_final)
        
        if st.button("Analyze Single File", type="primary", key="single_analyze"):
            with st.spinner("Processing audio file..."):
                path_single = "temp_single.wav" if hasattr(single_final, 'read') else single_final
                
                if hasattr(single_final, 'read'):
                    with open(path_single, "wb") as f:
                        f.write(single_final.getbuffer())
                
                y_single, sr_single, spec_single, _ = process_audio(path_single)
                
                # Forensic analysis
                rms = librosa.feature.rms(y=y_single)[0]
                thresh = np.mean(rms) * 0.3
                
                shimmer_curve = np.array([
                    np.std(rms[max(0,i-5):i+5])/(np.mean(rms[max(0,i-5):i+5])+1e-10)
                    for i in range(len(rms))
                ])
                
                chunk = y_single[:min(len(y_single), sr_single*4)]
                dct = np.abs(np.fft.rfft(chunk, n=8192))
                peaks, _ = find_peaks(dct, height=np.percentile(dct, 90), distance=10)
                
                # Display results
                st.markdown("---")
                st.markdown("## Analysis Results")
                
                # Get VoiceDetector prediction
                detector = get_voice_detector()
                detection_result = detector.predict(path_single)
                
                # Display detector verdict prominently
                det_label = detection_result["label"].upper()
                det_confidence = detection_result["confidence"]
                det_verdict = detection_result["verdict"]
                
                # Color based on label
                det_color = T["red"] if det_label == "FAKE" else T["green"]
                
                col_main1, col_main2 = st.columns([2, 1])
                
                with col_main1:
                    st.markdown(f"""
                    <div style="
                        background: {T['card']};
                        border-radius: 10px;
                        padding: 20px 24px;
                        border-left: 5px solid {det_color};
                    ">
                        <h2 style="margin: 0; color: {det_color}; letter-spacing: 0.04em;">{det_label}</h2>
                        <p style="color: {T['text_muted']}; margin-top: 6px; font-size: 0.9rem;">{det_verdict}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_main2:
                    st.metric("Confidence", f"{det_confidence*100:.1f}%")
                
                # Detector model scores
                st.markdown("#### Detector Model Scores")
                score_cols = st.columns(4)
                scores = detection_result["scores"]
                
                with score_cols[0]:
                    st.metric("Ensemble", f"{scores['ensemble']:.3f}")
                with score_cols[1]:
                    st.metric("Hyperparams", f"{scores['hyperparams']:.3f}")
                with score_cols[2]:
                    st.metric("XGBoost", f"{scores['xgboost']:.3f}" if scores['xgboost'] is not None else "N/A")
                with score_cols[3]:
                    st.metric("AASIST", f"{scores['aasist']:.3f}" if scores['aasist'] is not None else "N/A")
                
                # Display triggered signals if any
                if detection_result["signals"]:
                    st.markdown("#### Detection Signals Triggered")
                    for signal in detection_result["signals"][:8]:
                        st.warning(f"• {signal}")
                
                st.markdown("---")
                st.markdown("### Forensic Metrics")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                snr_ratio = np.mean(rms[rms>thresh])/(np.mean(rms[rms<=thresh])+1e-10)
                shim_val = np.mean(shimmer_curve)
                dct_score = len(peaks)
                
                with col_s1:
                    col_s1.metric("SNR Ratio", f"{snr_ratio:.2f}", "TRIGGER" if snr_ratio > 3 else "OK")
                
                with col_s2:
                    col_s2.metric("Shimmer Value", f"{shim_val:.4f}", "TRIGGER" if shim_val < 0.02 else "OK")
                
                with col_s3:
                    col_s3.metric("DCT Peaks", f"{dct_score}", "TRIGGER" if dct_score > 50 else "OK")
                
                # Spectrogram
                st.markdown("### Spectrogram")
                fig_spec, ax = plt.subplots(figsize=(12, 4), facecolor=T['card'])
                ax.set_facecolor(T['card'])
                librosa.display.specshow(spec_single, sr=sr_single, x_axis='time', ax=ax, cmap='viridis')
                ax.set_title("Audio Spectrogram", color=T['text'], fontsize=10, fontweight='bold')
                ax.tick_params(colors=T['text_muted'])
                st.pyplot(fig_spec, use_container_width=True)
                
                # Signal properties
                st.markdown("### Signal Properties")
                sig_col1, sig_col2 = st.columns(2)
                
                with sig_col1:
                    fig_rms = go.Figure()
                    fig_rms.add_trace(go.Scatter(y=rms, name="RMS Energy", fill='tozeroy', 
                                                  line=dict(color=T['green'], width=2)))
                    fig_rms.update_layout(template="plotly_dark", height=300, title="RMS Energy Over Time")
                    st.plotly_chart(fig_rms, use_container_width=True)
                
                with sig_col2:
                    fig_shim = go.Figure()
                    fig_shim.add_trace(go.Scatter(y=shimmer_curve, name="Shimmer", 
                                                   line=dict(color=T['purple'], width=2)))
                    fig_shim.update_layout(template="plotly_dark", height=300, title="Shimmer Value Over Time")
                    st.plotly_chart(fig_shim, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Compare Two Audio Files")
    st.caption("Compare a real voice with a generated/suspected clone")
    
    col_main1, col_btn, col_main2 = st.columns([4, 1, 4])

    with col_main1:
        st.markdown("### Real Audio")
        st.caption("Reference or original voice sample")
        c1, c2 = st.columns(2)
        with c1: real_upload = st.file_uploader("Upload New", type=['wav'], key="u1")
        with c2:
            real_local = get_local_files("input_voices")
            real_choice = st.selectbox("Select Server File", ["None"] + real_local, key="s1")

        real_final = real_upload if real_upload else (f"input_voices/{real_choice}" if real_choice!="None" else None)
        if real_final: 
            st.audio(real_final)

    with col_main2:
        st.markdown("### Generated Audio")
        st.caption("Voice clone or suspected synthetic sample")
        c3, c4 = st.columns(2)
        with c3: gen_upload = st.file_uploader("Upload New", type=['wav'], key="u2")
        with c4:
            gen_local = get_local_files("output_voices/xtts_finetune/shalini")
            gen_choice = st.selectbox("Select Server File", ["None"] + gen_local, key="s2")

        gen_final = gen_upload if gen_upload else (f"output_voices/xtts_finetune/shalini/{gen_choice}" if gen_choice!="None" else None)
        if gen_final: 
            st.audio(gen_final)

    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("ANALYZE", use_container_width=True, type="primary", key="analyze_btn")
    
    # COMPARISON ANALYSIS
    if analyze_clicked and real_final and gen_final:
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        with st.spinner("Processing audio files..."):
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

        # Verdict Logic (calculated but not displayed)
        verdict = (
            "HIGH MATCH" if cosine_sim > 0.8 and feature_dist < 5 else
            "PROBABLE CLONE" if cosine_sim > 0.6 else
            "DISTINCT VOICES"
        )

        # NOTE: The automated verdict card was removed from the UI because
        # it produced unreliable results. The `verdict` value is still
        # computed for logging/analysis purposes but will not be shown.

        # Key Metrics
        st.markdown("### Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Cosine Similarity", f"{cosine_sim*100:.1f}%", delta=f"{cosine_sim:.4f}", delta_color="inverse")
        
        with m2:
            st.metric("Feature Distance", f"{feature_dist:.2f}", delta="Lower is Better", delta_color="inverse")
        
        with m3:
            match_percent = min(100, int(cosine_sim * 100 + (100 - min(100, feature_dist * 10))))
            st.metric("Match Score", f"{match_percent}%")
        
        with m4:
            confidence = "High" if abs(cosine_sim - 0.8) < 0.2 or abs(cosine_sim - 0.6) < 0.2 else "Medium" if cosine_sim > 0.5 else "Low"
            st.metric("Confidence", confidence)

        # Spectrograms Comparison
        st.markdown("### Spectrogram Comparison")
        c_spec1, c_progress, c_spec2 = st.columns([2, 0.5, 2])

        with c_spec1:
            st.markdown("**Real Audio Spectrogram**")
            fig1, ax1 = plt.subplots(figsize=(6, 3), facecolor=T['card'])
            ax1.set_facecolor(T['card'])
            librosa.display.specshow(spec_r, sr=sr, x_axis='time', ax=ax1, cmap='viridis')
            ax1.set_title("Original Voice", color=T['text'], fontsize=10, fontweight='bold')
            ax1.tick_params(colors=T['text_muted'])
            st.pyplot(fig1, use_container_width=True)

        with c_progress:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            progress_val = float(min(1.0, cosine_sim))
            st.progress(progress_val)
            st.write("")
            st.metric("SIM", f"{cosine_sim:.2f}", label_visibility="collapsed")

        with c_spec2:
            st.markdown("**Generated Audio Spectrogram**")
            fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor=T['card'])
            ax2.set_facecolor(T['card'])
            librosa.display.specshow(spec_g, sr=sr, x_axis='time', ax=ax2, cmap='viridis')
            ax2.set_title("Generated/Clone Voice", color=T['text'], fontsize=10, fontweight='bold')
            ax2.tick_params(colors=T['text_muted'])
            st.pyplot(fig2, use_container_width=True)

        if show_detailed:
            st.markdown("---")
            st.markdown("### Forensic Parameter Analysis")
            
            tab_det1, tab_det2 = st.tabs(["Parameter Impact (34 Tiers)", "Signal Overlay"])
            
            with tab_det1:
                param_diffs = np.abs(feat_r - feat_g)
                param_names = (
                    [f"MFCC_Mean_{i}" for i in range(13)] +
                    [f"MFCC_Std_{i}" for i in range(13)] +
                    ["Centroid","Bandwidth","Flatness","ZCR","RMS_Mean","RMS_Std","Pitch_Mean","Pitch_Std"]
                )[:34]

                fig_bar = go.Figure(go.Bar(
                    x=param_diffs, 
                    y=param_names, 
                    orientation='h', 
                    marker=dict(color=param_diffs, colorscale='Viridis', showscale=True)
                ))
                fig_bar.update_layout(height=600, template="plotly_dark", title="Feature Parameter Differences")
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab_det2:
                fig_wave = go.Figure()
                step = max(1, len(y_r)//2000)
                fig_wave.add_trace(go.Scatter(y=y_r[::step], name="Real Voice", line=dict(color=T['red'], width=2)))
                fig_wave.add_trace(go.Scatter(y=y_g[::step], name="Generated Voice", line=dict(color=T['blue'], width=2)))
                fig_wave.update_layout(template="plotly_dark", height=400, title="Waveform Comparison")
                st.plotly_chart(fig_wave, use_container_width=True)

        if show_metrics:
            st.markdown("---")
            st.markdown("### Forensic Verdict Signals")
            
            rms = librosa.feature.rms(y=y_g)[0]
            thresh = np.mean(rms) * 0.3
            shimmer_curve = np.array([
                np.std(rms[max(0,i-5):i+5])/(np.mean(rms[max(0,i-5):i+5])+1e-10)
                for i in range(len(rms))
            ])
            chunk = y_g[:min(len(y_g), sr*4)]
            dct = np.abs(np.fft.rfft(chunk, n=8192))
            peaks, _ = find_peaks(dct, height=np.percentile(dct, 90), distance=10)
            
            snr_ratio = np.mean(rms[rms>thresh])/(np.mean(rms[rms<=thresh])+1e-10)
            shim_val = np.mean(shimmer_curve)
            dct_score = len(peaks)

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                col_m1.metric("Signal-to-Noise Ratio", f"{snr_ratio:.2f}", "TRIGGER" if snr_ratio > 3 else "OK")

            with col_m2:
                col_m2.metric("Shimmer Value", f"{shim_val:.4f}", "TRIGGER" if shim_val < 0.02 else "OK")

            with col_m3:
                col_m3.metric("DCT Peaks Detected", f"{dct_score}", "TRIGGER" if dct_score > 50 else "OK")

            st.markdown("---")
            st.markdown("### Forensic Interpretation")
            
            triggers = []
            
            if snr_ratio > 3:
                triggers.append("**SNR Triggered** → Silence regions are unnaturally clean (typical TTS artifact)")
            
            if shim_val < 0.02:
                triggers.append("**Shimmer Triggered** → Voice amplitude lacks natural variation (synthetic indicator)")
            
            if dct_score > 50:
                triggers.append("**DCT Triggered** → Periodic spectral peaks detected (upsampling/resampling artifact)")

            if triggers:
                st.error("**Synthetic Indicators Detected**\n\n" + "\n\n".join(triggers))
            else:
                st.success("**No strong synthetic artifacts detected.** Human review recommended for final confirmation.")
    
    elif analyze_clicked:
        st.warning("Please select both audio files before analyzing.")