"""
Audio preprocessing utilities for CompSpoofV2 dataset.
Handles audio loading, preprocessing, and spectrogram conversion.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
from pathlib import Path
import warnings


class AudioPreprocessor:
    """
    Audio preprocessing module for spoofing detection.
    Handles resampling, normalization, and spectrogram conversion.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        n_mels: int = 64,
        hop_length: int = 160,
        f_min: float = 80.0,
        f_max: float = 7600.0,
        normalize: bool = True,
        center: bool = True,
        pad_mode: str = "reflect",
        top_db: float = 60.0,
        ref: float = 1.0,
        amin: float = 1e-10,
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate (Hz)
            n_fft: FFT window size
            n_mels: Number of mel-frequency bins
            hop_length: Number of samples between successive frames
            f_min: Minimum frequency (Hz)
            f_max: Maximum frequency (Hz)
            normalize: Whether to normalize audio (z-score)
            center: Whether to center-pad the signal before windowing
            pad_mode: Padding mode for spectrograms
            top_db: Threshold in dB below reference
            ref: Reference power for dB computation
            amin: Minimum amplitude
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        self.center = center
        self.pad_mode = pad_mode
        self.top_db = top_db
        self.ref = ref
        self.amin = amin
        
        # Precompute mel filterbank for efficiency
        self.mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
    
    def load_audio(
        self,
        audio_path: Union[str, Path],
        target_duration: Optional[float] = None,
        pad_mode: str = "constant",
    ) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds (pad/trim if specified)
            pad_mode: Padding mode ('constant', 'repeat')
        
        Returns:
            Audio signal as numpy array (1D)
        
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If audio loading fails
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True,
                dtype=np.float32,
            )
            
            # Handle duration
            if target_duration is not None:
                target_samples = int(target_duration * self.sample_rate)
                
                if len(audio) < target_samples:
                    if pad_mode == "constant":
                        audio = np.pad(
                            audio,
                            (0, target_samples - len(audio)),
                            mode="constant",
                            constant_values=0,
                        )
                    elif pad_mode == "repeat":
                        repeats = (target_samples // len(audio)) + 1
                        audio = np.tile(audio, repeats)[:target_samples]
                else:
                    # Randomly crop if longer
                    start_idx = np.random.randint(0, len(audio) - target_samples + 1)
                    audio = audio[start_idx:start_idx + target_samples]
            
            return audio
        
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {audio_path}: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize audio signal.
        
        Args:
            audio: Audio signal (1D numpy array)
            method: Normalization method ('zscore', 'minmax', 'peak')
        
        Returns:
            Normalized audio signal
        """
        if method == "zscore":
            mean = np.mean(audio)
            std = np.std(audio)
            if std > 0:
                audio = (audio - mean) / (std + 1e-10)
        
        elif method == "minmax":
            min_val = np.min(audio)
            max_val = np.max(audio)
            if max_val - min_val > 0:
                audio = 2 * (audio - min_val) / (max_val - min_val) - 1
        
        elif method == "peak":
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / (peak + 1e-10)
        
        return audio.astype(np.float32)
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Audio signal (1D numpy array)
        
        Returns:
            STFT magnitude spectrogram (2D numpy array: freq x time)
        """
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=self.center,
            pad_mode=self.pad_mode,
        )
        
        # Compute magnitude
        magnitude = np.abs(stft)
        
        return magnitude.astype(np.float32)
    
    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        db_convert: bool = True,
        normalize_spec: bool = True,
    ) -> np.ndarray:
        """
        Compute Mel spectrogram.
        
        Args:
            audio: Audio signal (1D numpy array)
            db_convert: Convert to dB scale
            normalize_spec: Normalize spectrogram
        
        Returns:
            Mel spectrogram (2D numpy array: mel_freq x time)
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=self.center,
            pad_mode=self.pad_mode,
        )
        
        # Compute magnitude
        magnitude = np.abs(stft)
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_fb, magnitude)
        
        # Convert to dB
        if db_convert:
            mel_spec = librosa.power_to_db(
                mel_spec,
                ref=self.ref,
                amin=self.amin,
                top_db=self.top_db,
            )
        
        # Normalize spectrogram
        if normalize_spec:
            mean = np.mean(mel_spec)
            std = np.std(mel_spec)
            if std > 0:
                mel_spec = (mel_spec - mean) / (std + 1e-10)
        
        return mel_spec.astype(np.float32)
    
    def compute_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
    ) -> np.ndarray:
        """
        Compute MFCC (Mel-Frequency Cepstral Coefficients).
        
        Args:
            audio: Audio signal (1D numpy array)
            n_mfcc: Number of MFCCs
        
        Returns:
            MFCC features (2D numpy array: n_mfcc x time)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        
        return mfcc.astype(np.float32)
    
    def compute_chromagram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chromagram (12-bin chroma features).
        
        Args:
            audio: Audio signal (1D numpy array)
        
        Returns:
            Chromagram (2D numpy array: 12 x time)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        return chroma.astype(np.float32)
    
    def extract_features(
        self,
        audio: np.ndarray,
        feature_type: str = "mel",
    ) -> Union[np.ndarray, dict]:
        """
        Extract various audio features.
        
        Args:
            audio: Audio signal (1D numpy array)
            feature_type: Type of features ('mel', 'mfcc', 'stft', 'chroma', 'all')
        
        Returns:
            Feature array or dictionary of feature arrays if feature_type='all'
        """
        if feature_type == "mel":
            return self.compute_mel_spectrogram(audio)
        
        elif feature_type == "mfcc":
            return self.compute_mfcc(audio)
        
        elif feature_type == "stft":
            return self.compute_stft(audio)
        
        elif feature_type == "chroma":
            return self.compute_chromagram(audio)
        
        elif feature_type == "all":
            return {
                "mel": self.compute_mel_spectrogram(audio),
                "mfcc": self.compute_mfcc(audio),
                "stft": self.compute_stft(audio),
                "chroma": self.compute_chromagram(audio),
            }
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def audio_to_spectrogram(
        self,
        audio_path: Union[str, Path],
        target_duration: Optional[float] = None,
        feature_type: str = "mel",
        normalize_audio: bool = True,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Complete pipeline: load audio -> normalize -> compute spectrogram.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds
            feature_type: Type of spectrogram ('mel', 'mfcc', 'stft', 'chroma')
            normalize_audio: Whether to normalize audio before spectrogram
            return_tensor: Whether to return PyTorch tensor instead of numpy array
        
        Returns:
            Spectrogram as numpy array or PyTorch tensor
        """
        # Load audio
        audio = self.load_audio(audio_path, target_duration=target_duration)
        
        # Normalize audio
        if normalize_audio:
            audio = self.normalize_audio(audio)
        
        # Extract features
        spectrogram = self.extract_features(audio, feature_type=feature_type)
        
        # Convert to tensor if requested
        if return_tensor:
            if isinstance(spectrogram, dict):
                spectrogram = {k: torch.from_numpy(v) for k, v in spectrogram.items()}
            else:
                spectrogram = torch.from_numpy(spectrogram)
        
        return spectrogram


# Convenience function for quick access
def get_spectrogram(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    target_duration: Optional[float] = None,
    feature_type: str = "mel",
    return_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Quick function to get spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (Hz)
        n_mels: Number of mel-frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        target_duration: Target duration in seconds
        feature_type: Type of spectrogram
        return_tensor: Whether to return PyTorch tensor
    
    Returns:
        Spectrogram as numpy array or PyTorch tensor
    """
    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    
    return preprocessor.audio_to_spectrogram(
        audio_path=audio_path,
        target_duration=target_duration,
        feature_type=feature_type,
        return_tensor=return_tensor,
    )


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=64,
        n_fft=512,
        hop_length=160,
    )
    
    print("AudioPreprocessor initialized successfully!")
    print(f"Configuration:")
    print(f"  Sample Rate: {preprocessor.sample_rate} Hz")
    print(f"  N-FFT: {preprocessor.n_fft}")
    print(f"  Mel Bins: {preprocessor.n_mels}")
    print(f"  Hop Length: {preprocessor.hop_length}")
