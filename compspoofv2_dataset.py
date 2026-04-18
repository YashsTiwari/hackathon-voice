"""
PyTorch Dataset and DataLoader for CompSpoofV2 audio spoofing detection dataset.
Handles loading audio files, preprocessing, and spectrogram conversion.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from audio_preprocessing import AudioPreprocessor
import warnings


class CompSpoofV2Dataset(Dataset):
    """
    PyTorch Dataset for CompSpoofV2 spoofing detection dataset.
    
    Dataset structure:
    - Training: CompSpoofV2/development/metadata/train.csv
    - Evaluation: CompSpoofV2/eval/metadata/eval.csv
    - Audio files: paths specified in CSV
    
    Labels:
    - bonafide_bonafide: Genuine speech with genuine environment
    - spoof_bonafide: Spoofed/synthetic speech with genuine environment
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        dataset_root: Union[str, Path] = "./CompSpoofV2",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        target_duration: Optional[float] = 3.0,
        feature_type: str = "mel",
        normalize_audio: bool = True,
        normalize_spec: bool = True,
        split: Optional[str] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        cache_spectrograms: bool = False,
        num_workers_preprocessing: int = 1,
    ):
        """
        Initialize CompSpoofV2 Dataset.
        
        Args:
            csv_path: Path to metadata CSV file
            dataset_root: Root directory of the dataset
            sample_rate: Target sample rate (Hz)
            n_mels: Number of mel-frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            target_duration: Target duration in seconds (None = no padding/trimming)
            feature_type: Type of spectrogram ('mel', 'mfcc', 'stft', 'chroma')
            normalize_audio: Whether to normalize audio before spectrogram
            normalize_spec: Whether to normalize spectrogram
            split: Filter by split ('train', 'val', 'eval') - None means all
            label_mapping: Mapping from label strings to integers
            cache_spectrograms: Whether to cache spectrograms in memory
            num_workers_preprocessing: Number of workers for preprocessing (not used currently)
        """
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.feature_type = feature_type
        self.normalize_audio = normalize_audio
        self.normalize_spec = normalize_spec
        self.cache_spectrograms = cache_spectrograms
        self.split = split
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
        )
        
        # Default label mapping (5-class classification)
        # All classes in CompSpoofV2 dataset
        if label_mapping is None:
            self.label_mapping = {
                "original": 0,
                "bonafide_bonafide": 1,
                "spoof_bonafide": 2,
                "bonafide_spoof": 3,
                "spoof_spoof": 4,
            }
        else:
            self.label_mapping = label_mapping
        
        # Load metadata
        self.logger = logging.getLogger(__name__)
        self._load_metadata()
        
        # Cache for spectrograms
        self.spectrogram_cache = {} if cache_spectrograms else None
        
        self.logger.info(
            f"Initialized CompSpoofV2Dataset with {len(self)} samples"
        )
    
    def _load_metadata(self):
        """Load and validate metadata from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.logger.info(f"Loading metadata from {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {str(e)}")
        
        # Rename columns if needed for compatibility
        if 'audio_path' not in df.columns:
            if 'path' in df.columns:
                df = df.rename(columns={'path': 'audio_path'})
            else:
                raise ValueError("CSV must contain 'audio_path' or 'path' column")
        
        # Filter by split if specified
        if self.split is not None:
            if 'split' in df.columns:
                df = df[df['split'] == self.split].reset_index(drop=True)
                self.logger.info(
                    f"Filtered to split='{self.split}': {len(df)} samples"
                )
        
        # Filter by label
        if 'label' in df.columns:
            valid_labels = set(self.label_mapping.keys())
            df = df[df['label'].isin(valid_labels)].reset_index(drop=True)
            self.logger.info(
                f"After label filtering: {len(df)} samples"
            )
        
        self.metadata = df
        
        # Print label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            self.logger.info("Label distribution:")
            for label, count in label_counts.items():
                self.logger.info(f"  {label}: {count}")
        
        # Validate audio paths
        invalid_paths = []
        for idx, row in df.iterrows():
            audio_path = self.dataset_root / row['audio_path']
            if not audio_path.exists():
                invalid_paths.append((idx, row['audio_path']))
        
        if invalid_paths:
            self.logger.warning(
                f"Found {len(invalid_paths)} missing audio files. "
                f"First 5: {invalid_paths[:5]}"
            )
            # Filter out invalid paths
            valid_indices = [
                idx for idx in range(len(df))
                if (self.dataset_root / df.iloc[idx]['audio_path']).exists()
            ]
            self.metadata = df.iloc[valid_indices].reset_index(drop=True)
            self.logger.info(f"After removing invalid paths: {len(self.metadata)} samples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary with keys:
                - 'spectrogram': Spectrogram tensor (C, F, T) or (F, T)
                - 'label': Integer label (0 or 1)
                - 'label_name': String label name
                - 'audio_path': Path to audio file
                - (optional) additional metadata columns
        """
        row = self.metadata.iloc[idx]
        audio_path = self.dataset_root / row['audio_path']
        
        # Try to get from cache first
        if self.spectrogram_cache is not None and str(audio_path) in self.spectrogram_cache:
            spectrogram = self.spectrogram_cache[str(audio_path)]
        else:
            # Load and process audio
            try:
                spectrogram = self.preprocessor.audio_to_spectrogram(
                    audio_path=audio_path,
                    target_duration=self.target_duration,
                    feature_type=self.feature_type,
                    normalize_audio=self.normalize_audio,
                    return_tensor=True,
                )
                
                # Cache if enabled
                if self.spectrogram_cache is not None:
                    self.spectrogram_cache[str(audio_path)] = spectrogram
            
            except Exception as e:
                self.logger.error(
                    f"Error loading audio {audio_path}: {str(e)}"
                )
                # Return a zero spectrogram as fallback
                if self.feature_type == "mel":
                    h, w = 64, int(self.target_duration * self.sample_rate / 160) if self.target_duration else 100
                    spectrogram = torch.zeros((h, w), dtype=torch.float32)
                else:
                    spectrogram = torch.zeros((64, 100), dtype=torch.float32)
        
        # Get label
        label_name = row.get('label', 'unknown')
        label = self.label_mapping.get(label_name, -1)
        
        # Ensure spectrogram is 3D (add channel dimension if needed)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # (1, F, T)
        
        result = {
            'spectrogram': spectrogram,
            'label': label,
            'label_name': label_name,
            'audio_path': str(audio_path),
        }
        
        # Add additional metadata
        for col in self.metadata.columns:
            if col not in ['audio_path', 'label']:
                result[col] = row[col]
        
        return result
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset."""
        if 'label' not in self.metadata.columns:
            return {}
        return self.metadata['label'].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        if 'label' not in self.metadata.columns:
            return torch.ones(len(self.label_mapping))
        
        label_counts = self.metadata['label'].value_counts().sort_index()
        total_samples = len(self.metadata)
        
        weights = torch.tensor(
            [total_samples / count for count in label_counts.values],
            dtype=torch.float32
        )
        
        return weights / weights.sum() * len(weights)


def create_compspoofv2_dataloader(
    csv_path: Union[str, Path],
    dataset_root: Union[str, Path] = "./CompSpoofV2",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 160,
    target_duration: Optional[float] = 3.0,
    feature_type: str = "mel",
    normalize_audio: bool = True,
    normalize_spec: bool = True,
    split: Optional[str] = None,
    cache_spectrograms: bool = False,
) -> Tuple[DataLoader, CompSpoofV2Dataset]:
    """
    Create a PyTorch DataLoader for CompSpoofV2 dataset.
    
    Args:
        csv_path: Path to metadata CSV file
        dataset_root: Root directory of the dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for GPU transfer
        sample_rate: Target sample rate (Hz)
        n_mels: Number of mel-frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        target_duration: Target duration in seconds
        feature_type: Type of spectrogram
        normalize_audio: Whether to normalize audio
        normalize_spec: Whether to normalize spectrogram
        split: Filter by split
        cache_spectrograms: Whether to cache spectrograms
    
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = CompSpoofV2Dataset(
        csv_path=csv_path,
        dataset_root=dataset_root,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        target_duration=target_duration,
        feature_type=feature_type,
        normalize_audio=normalize_audio,
        normalize_spec=normalize_spec,
        split=split,
        cache_spectrograms=cache_spectrograms,
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length spectrograms."""
        # Get max length in batch
        max_time_steps = max(item['spectrogram'].shape[-1] for item in batch)
        
        # Pad spectrograms to max length
        padded_spectrograms = []
        labels = []
        label_names = []
        audio_paths = []
        
        for item in batch:
            spec = item['spectrogram']  # (C, F, T)
            
            # Pad to max_time_steps
            if spec.shape[-1] < max_time_steps:
                pad_amount = max_time_steps - spec.shape[-1]
                spec = torch.nn.functional.pad(spec, (0, pad_amount), mode='constant', value=0)
            
            padded_spectrograms.append(spec)
            labels.append(item['label'])
            label_names.append(item['label_name'])
            audio_paths.append(item['audio_path'])
        
        # Stack into batch
        batch_spectrograms = torch.stack(padded_spectrograms, dim=0)
        batch_labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'spectrogram': batch_spectrograms,
            'label': batch_labels,
            'label_name': label_names,
            'audio_path': audio_paths,
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset
    csv_path = "CompSpoofV2/development/metadata/train.csv"
    dataset = CompSpoofV2Dataset(
        csv_path=csv_path,
        dataset_root="CompSpoofV2",
        sample_rate=16000,
        n_mels=64,
        target_duration=3.0,
        feature_type="mel",
        split="train",
    )
    
    print(f"\nDataset created: {len(dataset)} samples")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    
    # Create dataloader
    dataloader, _ = create_compspoofv2_dataloader(
        csv_path=csv_path,
        dataset_root="CompSpoofV2",
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )
    
    print(f"\nDataLoader created with batch size: 32")
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"\nBatch shape:")
    print(f"  Spectrogram: {batch['spectrogram'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  First 3 labels: {batch['label'][:3].tolist()}")
