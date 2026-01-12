import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')
class DataConfig:
    NUM_CHANNELS = 22
    SAMPLE_RATE = 256
    TOTAL_FEATURES = 36864
    LABEL_INTERICTAL = 0
    LABEL_ICTAL = 1
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
class MITCHBDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, detection_labels: np.ndarray, forecast_labels: np.ndarray, transform: Optional[callable]=None, mode: str='train'):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.detection_labels = torch.LongTensor(detection_labels)
        self.forecast_labels = torch.LongTensor(forecast_labels)
        self.transform = transform
        self.mode = mode
        self.class_counts = np.bincount(labels.astype(int), minlength=2)
        self.class_weights = 1.0 / (self.class_counts + 1e-06)
        self.class_weights = self.class_weights / self.class_weights.sum()
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.data[idx]
        if self.transform is not None and self.mode == 'train':
            x = self.transform(x)
        return {'eeg': x, 'detection_label': self.detection_labels[idx], 'forecast_label': self.forecast_labels[idx], 'original_label': self.labels[idx], 'idx': idx}
    def get_sample_weights(self) -> torch.Tensor:
        weights = self.class_weights[self.labels.numpy()]
        return torch.FloatTensor(weights)
class EEGDataLoader:
    def __init__(self, data_path: str, config: DataConfig=DataConfig()):
        self.data_path = data_path
        self.config = config
        self.raw_data = None
        self.raw_labels = None
        self.processed_data = None
        self.cache_dir = os.path.dirname(data_path)
        self.features_cache = os.path.join(self.cache_dir, 'features_cache.npy')
        self.labels_cache = os.path.join(self.cache_dir, 'labels_cache.npy')
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.exists(self.features_cache) and os.path.exists(self.labels_cache):
            print('[INFO] Loading from cache (fast)...')
            sys.stdout.flush()
            features = np.load(self.features_cache)
            labels = np.load(self.labels_cache)
            print(f'[INFO] Loaded cached data: {features.shape}')
            self._print_label_distribution(labels)
            self.raw_data = features
            self.raw_labels = labels
            return (features, labels)
        print(f'[INFO] Loading data from: {self.data_path}')
        print('[INFO] First load takes 2-3 minutes for 2.5GB file...')
        print('[INFO] Subsequent loads will be instant from cache.')
        sys.stdout.flush()
        print('[INFO] Reading CSV (this is the slow part)...')
        sys.stdout.flush()
        chunk_size = 1000
        chunks = []
        total_rows = 11233
        reader = pd.read_csv(self.data_path, chunksize=chunk_size, dtype=np.float32, low_memory=False)
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            loaded = min((i + 1) * chunk_size, total_rows)
            pct = 100 * loaded / total_rows
            print(f'\r[INFO] Progress: {loaded}/{total_rows} rows ({pct:.0f}%)', end='')
            sys.stdout.flush()
        print()
        df = pd.concat(chunks, ignore_index=True)
        print(f'[INFO] Loaded DataFrame: {df.shape}')
        sys.stdout.flush()
        labels = df['target'].values.astype(np.int64)
        features = df.drop('target', axis=1).values.astype(np.float32)
        print('[INFO] Saving to cache for future fast loading...')
        np.save(self.features_cache, features)
        np.save(self.labels_cache, labels)
        print('[INFO] Cache saved!')
        self._print_label_distribution(labels)
        self.raw_data = features
        self.raw_labels = labels
        return (features, labels)
    def _print_label_distribution(self, labels):
        print(f'[INFO] Labels distribution:')
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f'       Class {u}: {c} samples ({100 * c / len(labels):.1f}%)')
        sys.stdout.flush()
    def reshape_to_channels(self, features: np.ndarray) -> np.ndarray:
        n_samples = features.shape[0]
        n_features = features.shape[1]
        n_channels = self.config.NUM_CHANNELS
        n_timesteps = n_features // n_channels
        usable_features = n_channels * n_timesteps
        print(f'[INFO] Reshaping: {n_features} -> ({n_channels} ch × {n_timesteps} time)')
        sys.stdout.flush()
        features_truncated = features[:, :usable_features]
        reshaped = features_truncated.reshape(n_samples, n_channels, n_timesteps)
        print(f'[INFO] Reshaped data: {reshaped.shape}')
        sys.stdout.flush()
        return reshaped
    def normalize(self, data: np.ndarray) -> np.ndarray:
        print('[INFO] Normalizing data...')
        sys.stdout.flush()
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = data.std(axis=(1, 2), keepdims=True) + 1e-08
        normalized = (data - mean) / std
        print('[INFO] Normalization complete')
        sys.stdout.flush()
        return normalized.astype(np.float32)
    def generate_forecast_labels(self, labels: np.ndarray, window_size: int=5) -> np.ndarray:
        forecast_labels = np.zeros_like(labels)
        seizure_indices = np.where(labels == 1)[0]
        for idx in seizure_indices:
            start_idx = max(0, idx - window_size)
            forecast_labels[start_idx:idx] = 1
        np.random.seed(self.config.RANDOM_SEED)
        n_additional = int(0.05 * len(seizure_indices))
        non_seizure_idx = np.where(labels == 0)[0]
        if len(non_seizure_idx) > n_additional:
            random_idx = np.random.choice(non_seizure_idx, size=n_additional, replace=False)
            forecast_labels[random_idx] = 1
        print(f'[INFO] Forecast labels: {sum(forecast_labels)} pre-ictal samples')
        sys.stdout.flush()
        return forecast_labels
    def create_splits(self, data: np.ndarray, labels: np.ndarray, detection_labels: np.ndarray, forecast_labels: np.ndarray) -> Dict[str, MITCHBDataset]:
        print('[INFO] Creating train/val/test splits...')
        sys.stdout.flush()
        train_val_idx, test_idx = train_test_split(np.arange(len(data)), test_size=self.config.TEST_RATIO, stratify=labels, random_state=self.config.RANDOM_SEED)
        val_ratio_adjusted = self.config.VAL_RATIO / (1 - self.config.TEST_RATIO)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio_adjusted, stratify=labels[train_val_idx], random_state=self.config.RANDOM_SEED)
        datasets = {'train': MITCHBDataset(data[train_idx], labels[train_idx], detection_labels[train_idx], forecast_labels[train_idx], mode='train'), 'val': MITCHBDataset(data[val_idx], labels[val_idx], detection_labels[val_idx], forecast_labels[val_idx], mode='val'), 'test': MITCHBDataset(data[test_idx], labels[test_idx], detection_labels[test_idx], forecast_labels[test_idx], mode='test')}
        print(f'\n[INFO] Dataset splits created:')
        for name, ds in datasets.items():
            seizure_count = int(sum(ds.detection_labels.numpy()))
            print(f'       {name}: {len(ds)} samples ({seizure_count} seizures)')
        sys.stdout.flush()
        return datasets
    def prepare_data(self) -> Dict[str, MITCHBDataset]:
        print('=' * 60)
        print('NEURALGRAPH DATA LOADING PIPELINE')
        print('=' * 60)
        sys.stdout.flush()
        features, labels = self.load_data()
        data = self.reshape_to_channels(features)
        data = self.normalize(data)
        detection_labels = labels.copy()
        forecast_labels = self.generate_forecast_labels(labels)
        datasets = self.create_splits(data, labels, detection_labels, forecast_labels)
        self.processed_data = data
        print('=' * 60)
        print('DATA LOADING COMPLETE')
        print('=' * 60)
        sys.stdout.flush()
        return datasets
def create_dataloaders(datasets: Dict[str, MITCHBDataset], batch_size: int=32, num_workers: int=0, use_weighted_sampling: bool=True) -> Dict[str, DataLoader]:
    dataloaders = {}
    for name, dataset in datasets.items():
        if name == 'train' and use_weighted_sampling:
            weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            dataloaders[name] = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, drop_last=True)
        else:
            dataloaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name == 'train', num_workers=num_workers, pin_memory=False, drop_last=False)
    return dataloaders
def visualize_spectrograms(dataset: MITCHBDataset, n_samples: int=4, save_path: Optional[str]=None):
    import matplotlib.pyplot as plt
    from scipy import signal
    channel_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8', 'T8-P8']
    seizure_idx = torch.where(dataset.detection_labels == 1)[0]
    normal_idx = torch.where(dataset.detection_labels == 0)[0]
    n_each = n_samples // 2
    selected_idx = []
    if len(seizure_idx) >= n_each:
        selected_idx.extend(seizure_idx[:n_each].tolist())
    if len(normal_idx) >= n_each:
        selected_idx.extend(normal_idx[:n_each].tolist())
    fig, axes = plt.subplots(len(selected_idx), 4, figsize=(16, 4 * len(selected_idx)))
    if len(selected_idx) == 1:
        axes = axes.reshape(1, -1)
    fs = 256
    for i, idx in enumerate(selected_idx):
        sample = dataset[idx]
        eeg = sample['eeg'].numpy()
        label = sample['detection_label'].item()
        for j, ch_idx in enumerate([0, 5, 10, 15]):
            if ch_idx < eeg.shape[0]:
                f, t, Sxx = signal.spectrogram(eeg[ch_idx], fs=fs, nperseg=64)
                ax = axes[i, j]
                im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                ax.set_ylabel('Freq (Hz)')
                ax.set_xlabel('Time (s)')
                ax.set_title(f"{channel_names[ch_idx]} - {('Seizure' if label == 1 else 'Normal')}")
                ax.set_ylim([0, 50])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Spectrogram saved to: {save_path}')
    plt.show()
class EEGVisualizer:
    def __init__(self, config: DataConfig=DataConfig()):
        self.config = config
        self.channel_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT10-T8', 'T8-P8']
    def plot_spectrograms(self, dataset: MITCHBDataset, n_samples: int=4, save_path: Optional[str]=None):
        visualize_spectrograms(dataset, n_samples, save_path)
    def plot_class_distribution(self, datasets: Dict[str, MITCHBDataset], save_path: Optional[str]=None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (name, dataset) in zip(axes, datasets.items()):
            labels = dataset.detection_labels.numpy()
            unique, counts = np.unique(labels, return_counts=True)
            colors = ['#2ecc71', '#e74c3c']
            ax.bar(unique, counts, color=colors)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'{name.capitalize()} Set')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Normal', 'Seizure'])
            for u, c in zip(unique, counts):
                ax.text(u, c + max(counts) * 0.02, str(c), ha='center', fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'[INFO] Class distribution saved to: {save_path}')
        plt.show()
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('TESTING DATASET LOADING')
    print('=' * 60 + '\n')
    data_path = 'mit-chb-subset/EEG_Scaled_data.csv'
    if not os.path.exists(data_path):
        print(f'[ERROR] Data file not found: {data_path}')
        sys.exit(1)
    loader = EEGDataLoader(data_path)
    datasets = loader.prepare_data()
    dataloaders = create_dataloaders(datasets, batch_size=32)
    print('\n[INFO] Testing batch loading...')
    for batch in dataloaders['train']:
        print(f"       Batch EEG shape: {batch['eeg'].shape}")
        print(f"       Batch labels: {batch['detection_label'][:5].tolist()}...")
        break
    print('\n[SUCCESS] Dataset loading test complete!')