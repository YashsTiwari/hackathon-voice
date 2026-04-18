"""
Data utilities and validation functions for CompSpoofV2 dataset.
Includes data analysis, validation, and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class CompSpoofV2Analyzer:
    """Analyzer for CompSpoofV2 dataset metadata and statistics."""
    
    def __init__(self, csv_path: Path, dataset_root: Path = None):
        """
        Initialize analyzer.
        
        Args:
            csv_path: Path to metadata CSV file
            dataset_root: Root directory of dataset (for path validation)
        """
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root) if dataset_root else Path(".")
        self.logger = logging.getLogger(__name__)
        
        # Load metadata
        self.df = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded metadata: {len(self.df)} samples")
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            raise
    
    def validate_paths(self) -> Dict[str, any]:
        """
        Validate that all audio paths exist.
        
        Returns:
            Dictionary with validation statistics
        """
        self.logger.info("Validating audio paths...")
        
        valid_paths = []
        invalid_paths = []
        
        for idx, row in self.df.iterrows():
            audio_path = self.dataset_root / row['audio_path']
            if audio_path.exists():
                valid_paths.append((idx, row['audio_path']))
            else:
                invalid_paths.append((idx, row['audio_path']))
        
        result = {
            'total': len(self.df),
            'valid': len(valid_paths),
            'invalid': len(invalid_paths),
            'valid_percentage': (len(valid_paths) / len(self.df)) * 100,
            'invalid_paths': invalid_paths,
        }
        
        self.logger.info(f"Path validation results:")
        self.logger.info(f"  Valid: {result['valid']}/{result['total']} ({result['valid_percentage']:.2f}%)")
        self.logger.info(f"  Invalid: {result['invalid']}/{result['total']}")
        
        return result
    
    def analyze_labels(self) -> Dict[str, any]:
        """
        Analyze label distribution.
        
        Returns:
            Dictionary with label statistics
        """
        if 'label' not in self.df.columns:
            self.logger.warning("'label' column not found in CSV")
            return {}
        
        label_counts = self.df['label'].value_counts()
        label_percentages = (label_counts / len(self.df)) * 100
        
        result = {
            'total_samples': len(self.df),
            'num_classes': len(label_counts),
            'label_distribution': label_counts.to_dict(),
            'label_percentages': label_percentages.to_dict(),
        }
        
        self.logger.info(f"Label distribution:")
        for label, count in label_counts.items():
            percentage = label_percentages[label]
            self.logger.info(f"  {label}: {count} ({percentage:.2f}%)")
        
        # Calculate class imbalance ratio
        max_count = label_counts.max()
        min_count = label_counts.min()
        result['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
        self.logger.info(f"Class imbalance ratio: {result['imbalance_ratio']:.2f}")
        
        return result
    
    def analyze_splits(self) -> Dict[str, any]:
        """
        Analyze split distribution if 'split' column exists.
        
        Returns:
            Dictionary with split statistics
        """
        if 'split' not in self.df.columns:
            self.logger.warning("'split' column not found in CSV")
            return {}
        
        split_counts = self.df['split'].value_counts()
        
        result = {'split_distribution': split_counts.to_dict()}
        
        self.logger.info(f"Split distribution:")
        for split, count in split_counts.items():
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"  {split}: {count} ({percentage:.2f}%)")
        
        # Analyze label distribution per split
        if 'label' in self.df.columns:
            cross_tab = pd.crosstab(self.df['split'], self.df['label'])
            result['split_label_distribution'] = cross_tab.to_dict()
            
            self.logger.info(f"Label distribution per split:")
            for split in cross_tab.index:
                self.logger.info(f"  {split}:")
                for label in cross_tab.columns:
                    count = cross_tab.loc[split, label]
                    self.logger.info(f"    {label}: {count}")
        
        return result
    
    def analyze_metadata_columns(self) -> Dict[str, any]:
        """
        Analyze all metadata columns.
        
        Returns:
            Dictionary with metadata column information
        """
        result = {
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
        }
        
        self.logger.info(f"Metadata columns:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            missing = self.df[col].isnull().sum()
            self.logger.info(f"  {col}: {dtype} (missing: {missing})")
        
        return result
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the dataset."""
        stats = {
            'metadata': self.analyze_metadata_columns(),
            'labels': self.analyze_labels(),
            'splits': self.analyze_splits(),
            'paths': self.validate_paths(),
        }
        
        return stats
    
    def plot_label_distribution(self, save_path: Optional[Path] = None):
        """
        Plot label distribution.
        
        Args:
            save_path: Path to save the plot
        """
        if 'label' not in self.df.columns:
            self.logger.warning("Cannot plot: 'label' column not found")
            return
        
        plt.figure(figsize=(8, 6))
        label_counts = self.df['label'].value_counts()
        plt.bar(range(len(label_counts)), label_counts.values)
        plt.xticks(range(len(label_counts)), label_counts.index)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_split_distribution(self, save_path: Optional[Path] = None):
        """
        Plot split distribution.
        
        Args:
            save_path: Path to save the plot
        """
        if 'split' not in self.df.columns:
            self.logger.warning("Cannot plot: 'split' column not found")
            return
        
        plt.figure(figsize=(8, 6))
        split_counts = self.df['split'].value_counts()
        plt.bar(range(len(split_counts)), split_counts.values)
        plt.xticks(range(len(split_counts)), split_counts.index)
        plt.xlabel('Split')
        plt.ylabel('Count')
        plt.title('Split Distribution')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_cross_validation(self, save_path: Optional[Path] = None):
        """
        Plot cross-validation between splits and labels.
        
        Args:
            save_path: Path to save the plot
        """
        if 'split' not in self.df.columns or 'label' not in self.df.columns:
            self.logger.warning("Cannot plot: 'split' and/or 'label' columns not found")
            return
        
        cross_tab = pd.crosstab(self.df['split'], self.df['label'])
        
        plt.figure(figsize=(10, 6))
        cross_tab.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Split')
        plt.ylabel('Count')
        plt.title('Label Distribution per Split')
        plt.legend(title='Label')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.close()


def validate_dataset_integrity(dataset_root: Path, csv_path: Path) -> bool:
    """
    Validate dataset integrity.
    
    Args:
        dataset_root: Root directory of dataset
        csv_path: Path to metadata CSV
    
    Returns:
        True if dataset is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    analyzer = CompSpoofV2Analyzer(csv_path, dataset_root)
    
    # Run validation
    path_stats = analyzer.validate_paths()
    label_stats = analyzer.analyze_labels()
    split_stats = analyzer.analyze_splits()
    
    # Check validity
    is_valid = True
    
    if path_stats['invalid'] > 0:
        logger.error(f"Found {path_stats['invalid']} invalid audio paths!")
        is_valid = False
    
    if label_stats and label_stats.get('imbalance_ratio', 1) > 5:
        logger.warning(f"High class imbalance: {label_stats['imbalance_ratio']:.2f}")
    
    return is_valid


def create_train_val_split(
    csv_path: Path,
    output_dir: Path,
    val_split: float = 0.2,
    random_state: int = 42,
):
    """
    Create train/val split from dataset.
    
    Args:
        csv_path: Path to metadata CSV
        output_dir: Output directory for new CSVs
        val_split: Validation split ratio
        random_state: Random state for reproducibility
    """
    logger = logging.getLogger(__name__)
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    
    # Split by label to maintain label distribution
    dfs = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        n_val = int(len(label_df) * val_split)
        
        label_df_shuffled = label_df.sample(frac=1, random_state=random_state)
        val_df = label_df_shuffled.iloc[:n_val].copy()
        train_df = label_df_shuffled.iloc[n_val:].copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        
        dfs.append(train_df)
        dfs.append(val_df)
    
    # Combine and save
    output_df = pd.concat(dfs, ignore_index=True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'train_val_split.csv'
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output_df)} samples to {output_path}")
    
    # Print statistics
    train_count = len(output_df[output_df['split'] == 'train'])
    val_count = len(output_df[output_df['split'] == 'val'])
    logger.info(f"Train: {train_count}, Val: {val_count} ({val_split*100:.1f}%)")
    
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    csv_path = Path("CompSpoofV2/development/metadata/train.csv")
    dataset_root = Path("CompSpoofV2")
    
    analyzer = CompSpoofV2Analyzer(csv_path, dataset_root)
    
    # Get statistics
    stats = analyzer.get_statistics()
    
    # Create plots
    output_dir = Path("./analysis_plots")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.plot_label_distribution(output_dir / "label_distribution.png")
    analyzer.plot_split_distribution(output_dir / "split_distribution.png")
    analyzer.plot_cross_validation(output_dir / "split_label_cross.png")
    
    print("\nAnalysis complete!")
