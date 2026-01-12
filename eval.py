import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_simple import NeuralGraphSimple, count_parameters, get_model_size_mb
class Config:
    CACHE_DIR = 'mit-chb-subset'
    NUM_CHANNELS = 22
    HIDDEN_DIM = 128
    USE_GRAPH = False
    BATCH_SIZE = 32
    TEST_RATIO = 0.2
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = 'checkpoints'
def load_test_data(config):
    features = np.load(os.path.join(config.CACHE_DIR, 'features_cache.npy'))
    labels = np.load(os.path.join(config.CACHE_DIR, 'labels_cache.npy'))
    n_samples = features.shape[0]
    time_steps = features.shape[1] // config.NUM_CHANNELS
    usable = time_steps * config.NUM_CHANNELS
    features = features[:, :usable].reshape(n_samples, config.NUM_CHANNELS, time_steps)
    mean = features.mean(axis=2, keepdims=True)
    std = features.std(axis=2, keepdims=True) + 1e-08
    features = (features - mean) / std
    _, X_test, _, y_test = train_test_split(features, labels, test_size=config.TEST_RATIO, random_state=config.SEED, stratify=labels)
    return (X_test, y_test, time_steps)
def load_model(config, time_steps):
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model = NeuralGraphSimple(num_channels=config.NUM_CHANNELS, time_steps=time_steps, hidden_dim=config.HIDDEN_DIM, use_graph=config.USE_GRAPH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    return (model, checkpoint)
def evaluate_model(model, X_test, y_test, config, threshold=0.3):
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Evaluating'):
            data = data.to(config.DEVICE)
            outputs = model(data)
            probs = outputs['probs'][:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    results = {'threshold': threshold, 'accuracy': (tp + tn) / (tp + tn + fp + fn), 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp), 'precision': tp / (tp + fp) if tp + fp > 0 else 0, 'f1': 2 * tp / (2 * tp + fp + fn), 'auc_roc': roc_auc_score(all_labels, all_probs), 'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'total_samples': len(all_labels), 'seizure_samples': int(sum(all_labels)), 'normal_samples': int(len(all_labels) - sum(all_labels))}
    return (results, all_probs, all_labels)
def plot_results(results, probs, labels, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {results['auc_roc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - NeuralGraph Seizure Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2)
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150)
    plt.close()
    cm = np.array([[results['tn'], results['fp']], [results['fn'], results['tp']]])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (threshold={results['threshold']})", fontsize=14)
    plt.colorbar()
    classes = ['Non-Seizure', 'Seizure']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.hist(probs[labels == 0], bins=50, alpha=0.7, label='Non-Seizure', color='blue')
    plt.hist(probs[labels == 1], bins=50, alpha=0.7, label='Seizure', color='red')
    plt.axvline(x=results['threshold'], color='black', linestyle='--', linewidth=2, label=f"Threshold={results['threshold']}")
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_distribution.png'), dpi=150)
    plt.close()
    print(f'[INFO] Plots saved to {save_dir}/')
def main():
    parser = argparse.ArgumentParser(description='Evaluate NeuralGraph')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    config = Config()
    print('=' * 60)
    print('NEURALGRAPH EVALUATION')
    print('=' * 60)
    print(f'Device: {config.DEVICE}')
    print('\n[INFO] Loading test data...')
    X_test, y_test, time_steps = load_test_data(config)
    print(f'[INFO] Test samples: {len(X_test)} ({sum(y_test)} seizures)')
    print('\n[INFO] Loading model...')
    model, checkpoint = load_model(config, time_steps)
    print(f'[INFO] Model parameters: {count_parameters(model):,}')
    print(f'[INFO] Model size: {get_model_size_mb(model):.2f} MB')
    if 'optimal_threshold' in checkpoint.get('metrics', {}):
        recommended_threshold = checkpoint['metrics']['optimal_threshold']
        print(f'[INFO] Recommended threshold from training: {recommended_threshold}')
    print(f'\n[INFO] Evaluating with threshold={args.threshold}...')
    results, probs, labels = evaluate_model(model, X_test, y_test, config, args.threshold)
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f"\n  Threshold:     {results['threshold']}")
    print(f"\n  Accuracy:      {results['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity:   {results['sensitivity'] * 100:.2f}% (Seizures detected)")
    print(f"  Specificity:   {results['specificity'] * 100:.2f}% (Non-seizures correct)")
    print(f"  Precision:     {results['precision'] * 100:.2f}%")
    print(f"  F1-Score:      {results['f1']:.4f}")
    print(f"  AUC-ROC:       {results['auc_roc']:.4f}")
    print(f'\n  Confusion Matrix:')
    print(f"    True Negatives:  {results['tn']} (correct non-seizure)")
    print(f"    True Positives:  {results['tp']} (correct seizure)")
    print(f"    False Positives: {results['fp']} (false alarm)")
    print(f"    False Negatives: {results['fn']} (missed seizure)")
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[INFO] Results saved to outputs/evaluation_results.json')
    if not args.no_plots:
        print('\n[INFO] Generating plots...')
        plot_results(results, probs, labels)
    print('\n' + '=' * 60)
    print('EVALUATION COMPLETE')
    print('=' * 60)
if __name__ == '__main__':
    main()