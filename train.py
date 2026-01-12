import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from model_simple import NeuralGraphSimple, count_parameters, get_model_size_mb
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
class Config:
    DATA_PATH = 'mit-chb-subset/EEG_Scaled_data.csv'
    CACHE_DIR = 'mit-chb-subset'
    NUM_CHANNELS = 22
    TIME_STEPS = 1675
    HIDDEN_DIM = 128
    USE_GRAPH = False
    BATCH_SIZE = 32
    EPOCHS = 40
    LR = 0.0005
    WEIGHT_DECAY = 0.001
    OVERSAMPLE_RATIO = 3.0
    DECISION_THRESHOLD = 0.3
    TEST_RATIO = 0.2
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
def load_data(config: Config):
    features_cache = os.path.join(config.CACHE_DIR, 'features_cache.npy')
    labels_cache = os.path.join(config.CACHE_DIR, 'labels_cache.npy')
    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        print('[INFO] Loading from cache...')
        features = np.load(features_cache)
        labels = np.load(labels_cache)
    else:
        print('[INFO] Loading from CSV...')
        import pandas as pd
        df = pd.read_csv(config.DATA_PATH)
        labels = df.iloc[:, -1].values
        features = df.iloc[:, :-1].values
        np.save(features_cache, features)
        np.save(labels_cache, labels)
    print(f'[INFO] Data shape: {features.shape}')
    print(f'[INFO] Class distribution: {np.bincount(labels.astype(int))}')
    n_samples = features.shape[0]
    time_steps = features.shape[1] // config.NUM_CHANNELS
    usable = time_steps * config.NUM_CHANNELS
    features = features[:, :usable].reshape(n_samples, config.NUM_CHANNELS, time_steps)
    mean = features.mean(axis=2, keepdims=True)
    std = features.std(axis=2, keepdims=True) + 1e-08
    features = (features - mean) / std
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=config.TEST_RATIO, random_state=config.SEED, stratify=labels)
    print(f'\n[INFO] Train: {len(X_train)} ({(y_train == 1).sum()} seizures, {(y_train == 1).mean() * 100:.1f}%)')
    print(f'[INFO] Test:  {len(X_test)} ({(y_test == 1).sum()} seizures, {(y_test == 1).mean() * 100:.1f}%)')
    return (X_train, X_test, y_train, y_test)
def create_dataloaders(X_train, X_test, y_train, y_test, config: Config):
    class_counts = np.bincount(y_train.astype(int))
    weights = np.zeros(len(y_train))
    weights[y_train == 0] = 1.0
    weights[y_train == 1] = class_counts[0] / class_counts[1] * config.OVERSAMPLE_RATIO
    sampler = WeightedRandomSampler(weights=torch.FloatTensor(weights), num_samples=len(weights), replacement=True)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return (train_loader, test_loader)
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = ([], [])
    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False)
    for data, labels in pbar:
        data, labels = (data.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return (total_loss / len(loader), accuracy_score(all_labels, all_preds))
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_probs, all_labels = ([], [])
    with torch.no_grad():
        for data, labels in loader:
            data, labels = (data.to(device), labels.to(device))
            outputs = model(data)
            loss = criterion(outputs['logits'], labels)
            total_loss += loss.item()
            probs = outputs['probs'][:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    metrics = {'loss': total_loss / len(loader), 'accuracy': (tp + tn) / (tp + tn + fp + fn), 'sensitivity': tp / (tp + fn + 1e-08), 'specificity': tn / (tn + fp + 1e-08), 'precision': tp / (tp + fp + 1e-08), 'f1': 2 * tp / (2 * tp + fp + fn + 1e-08), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'threshold': threshold}
    try:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc_roc'] = 0.5
    return (metrics, all_probs, all_labels)
def find_optimal_threshold(probs, labels, target_sensitivity=0.9):
    best_threshold = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-08)
        specificity = tn / (tn + fp + 1e-08)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-08)
        if sensitivity >= target_sensitivity and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold
def train(config: Config=Config()):
    print('=' * 60)
    print('NEURALGRAPH BALANCED TRAINING')
    print('=' * 60)
    print(f'Device: {config.DEVICE}')
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    print('\n' + '-' * 40)
    X_train, X_test, y_train, y_test = load_data(config)
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, config)
    config.TIME_STEPS = X_train.shape[2]
    print('\n' + '-' * 40)
    print('Creating Model')
    model = NeuralGraphSimple(num_channels=config.NUM_CHANNELS, time_steps=config.TIME_STEPS, hidden_dim=config.HIDDEN_DIM, use_graph=config.USE_GRAPH).to(config.DEVICE)
    print(f'Parameters: {count_parameters(model):,}')
    print(f'Size: {get_model_size_mb(model):.2f} MB')
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    print('\n' + '-' * 40)
    print('Training')
    print('-' * 40)
    best_score = 0
    best_metrics = None
    history = []
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, epoch)
        test_metrics, test_probs, test_labels = evaluate(model, test_loader, criterion, config.DEVICE, threshold=config.DECISION_THRESHOLD)
        opt_threshold = find_optimal_threshold(test_probs, test_labels, target_sensitivity=0.85)
        opt_metrics, _, _ = evaluate(model, test_loader, criterion, config.DEVICE, threshold=opt_threshold)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        balanced_score = np.sqrt(test_metrics['sensitivity'] * test_metrics['specificity'])
        print(f'\nEpoch {epoch:2d}/{config.EPOCHS}')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}%')
        print(f"  Test @ {config.DECISION_THRESHOLD}: Sens={test_metrics['sensitivity'] * 100:.1f}% Spec={test_metrics['specificity'] * 100:.1f}% F1={test_metrics['f1']:.3f}")
        print(f"  Optimal @ {opt_threshold:.2f}: Sens={opt_metrics['sensitivity'] * 100:.1f}% Spec={opt_metrics['specificity'] * 100:.1f}% F1={opt_metrics['f1']:.3f}")
        print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f} | LR: {lr:.2e}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_sens': test_metrics['sensitivity'], 'test_spec': test_metrics['specificity'], 'test_f1': test_metrics['f1'], 'test_auc': test_metrics['auc_roc'], 'opt_threshold': opt_threshold, 'opt_sens': opt_metrics['sensitivity'], 'opt_spec': opt_metrics['specificity'], 'opt_f1': opt_metrics['f1']})
        if balanced_score > best_score:
            best_score = balanced_score
            best_metrics = test_metrics.copy()
            best_metrics['optimal_threshold'] = opt_threshold
            best_metrics['optimal_sens'] = opt_metrics['sensitivity']
            best_metrics['optimal_spec'] = opt_metrics['specificity']
            best_metrics['optimal_f1'] = opt_metrics['f1']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'metrics': best_metrics, 'optimal_threshold': opt_threshold, 'config': {'num_channels': config.NUM_CHANNELS, 'time_steps': config.TIME_STEPS, 'hidden_dim': config.HIDDEN_DIM, 'use_graph': config.USE_GRAPH}}, os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
            print(f'  *** Best model saved! Balanced={balanced_score:.4f} ***')
    with open(os.path.join(config.LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'\nBest Metrics (threshold={config.DECISION_THRESHOLD}):')
    print(f"  Sensitivity: {best_metrics['sensitivity'] * 100:.1f}%")
    print(f"  Specificity: {best_metrics['specificity'] * 100:.1f}%")
    print(f"  F1-Score:    {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {best_metrics['auc_roc']:.4f}")
    print(f"\nWith Optimal Threshold ({best_metrics['optimal_threshold']:.2f}):")
    print(f"  Sensitivity: {best_metrics['optimal_sens'] * 100:.1f}%")
    print(f"  Specificity: {best_metrics['optimal_spec'] * 100:.1f}%")
    print(f"  F1-Score:    {best_metrics['optimal_f1']:.4f}")
    return (model, best_metrics, history)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--oversample', type=float, default=3.0)
    args = parser.parse_args()
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.DECISION_THRESHOLD = args.threshold
    config.OVERSAMPLE_RATIO = args.oversample
    train(config)