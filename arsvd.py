import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import copy
from collections import OrderedDict
class ApproximateRandomizedSVD:
    @staticmethod
    def randomized_svd(matrix: np.ndarray, rank: int, n_oversamples: int=10, n_power_iterations: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = matrix.shape
        k = min(rank + n_oversamples, min(m, n))
        np.random.seed(42)
        omega = np.random.randn(n, k)
        Y = matrix @ omega
        for _ in range(n_power_iterations):
            Y = matrix @ (matrix.T @ Y)
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ matrix
        U_b, S, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_b
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        return (U, S, Vt)
    @staticmethod
    def randomized_svd_torch(matrix: torch.Tensor, rank: int, n_oversamples: int=10, n_power_iterations: int=2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = matrix.device
        dtype = matrix.dtype
        m, n = matrix.shape
        k = min(rank + n_oversamples, min(m, n))
        torch.manual_seed(42)
        omega = torch.randn(n, k, device=device, dtype=dtype)
        Y = matrix @ omega
        for _ in range(n_power_iterations):
            Y = matrix @ (matrix.T @ Y)
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ matrix
        U_b, S, Vt = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_b
        return (U[:, :rank], S[:rank], Vt[:rank, :])
class LowRankLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, bias: Optional[torch.Tensor]=None):
        super().__init__()
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)
        self.Vt = nn.Parameter(Vt)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        self.rank = S.shape[0]
        self.in_features = Vt.shape[1]
        self.out_features = U.shape[0]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.Vt.T
        x = x * self.S
        x = x @ self.U.T
        if self.bias is not None:
            x = x + self.bias
        return x
    def get_full_weight(self) -> torch.Tensor:
        return self.U @ torch.diag(self.S) @ self.Vt
class LowRankConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, rank: int, stride: int=1, padding: int=0, dilation: int=1, bias: bool=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.conv1 = nn.Conv1d(in_channels, rank, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.conv2 = nn.Conv1d(rank, out_channels, 1, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    @classmethod
    def from_conv1d(cls, conv: nn.Conv1d, rank: int) -> 'LowRankConv1d':
        device = conv.weight.device
        out_ch, in_ch, k = conv.weight.shape
        weight_2d = conv.weight.data.view(out_ch, -1).cpu().numpy()
        U, S, Vt = ApproximateRandomizedSVD.randomized_svd(weight_2d, rank)
        sqrt_S = np.sqrt(S)
        U_scaled = U * sqrt_S[np.newaxis, :]
        Vt_scaled = Vt * sqrt_S[:, np.newaxis]
        low_rank_conv = cls(in_ch, out_ch, k, rank, stride=conv.stride[0], padding=conv.padding[0], dilation=conv.dilation[0], bias=conv.bias is not None)
        V_reshaped = Vt_scaled.reshape(rank, in_ch, k)
        low_rank_conv.conv1.weight.data = torch.FloatTensor(V_reshaped).to(device)
        low_rank_conv.conv2.weight.data = torch.FloatTensor(U_scaled).unsqueeze(-1).to(device)
        if conv.bias is not None:
            low_rank_conv.conv2.bias.data = conv.bias.data.clone()
        return low_rank_conv.to(device)
class ModelCompressor:
    def __init__(self, rank_ratio: float=0.5, min_rank: int=4):
        self.rank_ratio = rank_ratio
        self.min_rank = min_rank
        self.compression_stats = {}
    def compress_linear(self, layer: nn.Linear, name: str) -> LowRankLinear:
        device = layer.weight.device
        weight = layer.weight.data.cpu().numpy()
        out_features, in_features = weight.shape
        max_rank = min(out_features, in_features)
        target_rank = max(self.min_rank, int(max_rank * self.rank_ratio))
        U, S, Vt = ApproximateRandomizedSVD.randomized_svd(weight, target_rank)
        U = torch.FloatTensor(U).to(device)
        S = torch.FloatTensor(S).to(device)
        Vt = torch.FloatTensor(Vt).to(device)
        bias = layer.bias.data.clone() if layer.bias is not None else None
        compressed = LowRankLinear(U, S, Vt, bias)
        compressed = compressed.to(device)
        original_params = out_features * in_features
        compressed_params = target_rank * (out_features + in_features)
        if layer.bias is not None:
            original_params += out_features
            compressed_params += out_features
        compression_ratio = original_params / compressed_params
        self.compression_stats[name] = {'original_shape': (out_features, in_features), 'rank': target_rank, 'original_params': original_params, 'compressed_params': compressed_params, 'compression_ratio': compression_ratio}
        return compressed
    def compress_conv1d(self, layer: nn.Conv1d, name: str) -> LowRankConv1d:
        out_ch, in_ch, k = layer.weight.shape
        max_rank = min(out_ch, in_ch * k)
        target_rank = max(self.min_rank, int(max_rank * self.rank_ratio))
        compressed = LowRankConv1d.from_conv1d(layer, target_rank)
        original_params = out_ch * in_ch * k
        compressed_params = target_rank * (in_ch * k + out_ch)
        if layer.bias is not None:
            original_params += out_ch
            compressed_params += out_ch
        compression_ratio = original_params / compressed_params
        self.compression_stats[name] = {'original_shape': (out_ch, in_ch, k), 'rank': target_rank, 'original_params': original_params, 'compressed_params': compressed_params, 'compression_ratio': compression_ratio}
        return compressed
    def compress_model(self, model: nn.Module, exclude_layers: Optional[List[str]]=None) -> nn.Module:
        exclude_layers = exclude_layers or []
        compressed_model = copy.deepcopy(model)
        def compress_module(module: nn.Module, prefix: str=''):
            for name, child in module.named_children():
                full_name = f'{prefix}.{name}' if prefix else name
                if any((excl in full_name for excl in exclude_layers)):
                    continue
                if isinstance(child, nn.Linear):
                    if child.weight.numel() > 1000:
                        compressed = self.compress_linear(child, full_name)
                        setattr(module, name, compressed)
                elif isinstance(child, nn.Conv1d):
                    if child.weight.numel() > 500:
                        compressed = self.compress_conv1d(child, full_name)
                        setattr(module, name, compressed)
                else:
                    compress_module(child, full_name)
        compress_module(compressed_model)
        return compressed_model
    def get_compression_summary(self) -> Dict:
        total_original = sum((s['original_params'] for s in self.compression_stats.values()))
        total_compressed = sum((s['compressed_params'] for s in self.compression_stats.values()))
        return {'layers_compressed': len(self.compression_stats), 'total_original_params': total_original, 'total_compressed_params': total_compressed, 'overall_compression_ratio': total_original / max(total_compressed, 1), 'per_layer_stats': self.compression_stats}
    def print_compression_report(self):
        summary = self.get_compression_summary()
        print('\n' + '=' * 60)
        print('ARSVD COMPRESSION REPORT')
        print('=' * 60)
        print(f"\nLayers Compressed: {summary['layers_compressed']}")
        print(f"Original Parameters: {summary['total_original_params']:,}")
        print(f"Compressed Parameters: {summary['total_compressed_params']:,}")
        print(f"Overall Compression Ratio: {summary['overall_compression_ratio']:.2f}x")
        print('\n' + '-' * 60)
        print('Per-Layer Details:')
        print('-' * 60)
        for name, stats in self.compression_stats.items():
            print(f'\n{name}:')
            print(f"  Original Shape: {stats['original_shape']}")
            print(f"  Target Rank: {stats['rank']}")
            print(f"  Parameters: {stats['original_params']:,} -> {stats['compressed_params']:,}")
            print(f"  Compression: {stats['compression_ratio']:.2f}x")
def count_parameters(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters()))
def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum((p.numel() * p.element_size() for p in model.parameters()))
    buffer_size = sum((b.numel() * b.element_size() for b in model.buffers()))
    return (param_size + buffer_size) / 1024 ** 2
def compress_model(model: nn.Module, rank_ratio: float=0.5, exclude_layers: Optional[List[str]]=None) -> Tuple[nn.Module, Dict]:
    compressor = ModelCompressor(rank_ratio=rank_ratio)
    compressed = compressor.compress_model(model, exclude_layers)
    original_params = count_parameters(model)
    compressed_params = count_parameters(compressed)
    stats = {'original_params': original_params, 'compressed_params': compressed_params, 'compression_ratio': original_params / compressed_params, 'original_size_mb': get_model_size_mb(model), 'compressed_size_mb': get_model_size_mb(compressed), 'layer_stats': compressor.get_compression_summary()}
    return (compressed, stats)
def evaluate_approximation_error(original: nn.Module, compressed: nn.Module, test_input: torch.Tensor) -> Dict:
    original.eval()
    compressed.eval()
    with torch.no_grad():
        original_out = original(test_input)
        compressed_out = compressed(test_input)
        if isinstance(original_out, dict):
            orig_logits = original_out['detection_logits']
            comp_logits = compressed_out['detection_logits']
        else:
            orig_logits = original_out
            comp_logits = compressed_out
        mse = torch.mean((orig_logits - comp_logits) ** 2).item()
        mae = torch.mean(torch.abs(orig_logits - comp_logits)).item()
        max_error = torch.max(torch.abs(orig_logits - comp_logits)).item()
        rel_error = torch.norm(orig_logits - comp_logits) / torch.norm(orig_logits)
        cos_sim = torch.nn.functional.cosine_similarity(orig_logits.flatten().unsqueeze(0), comp_logits.flatten().unsqueeze(0)).item()
    return {'mse': mse, 'mae': mae, 'max_error': max_error, 'relative_error': rel_error.item(), 'cosine_similarity': cos_sim}
if __name__ == '__main__':
    '\n    Test ARSVD compression on sample models.\n    '
    import sys
    sys.path.insert(0, '.')
    print('=' * 60)
    print('NeuralGraph ARSVD Compression Test')
    print('=' * 60)
    print('\n[TEST 1] Randomized SVD on random matrix...')
    np.random.seed(42)
    A = np.random.randn(100, 50)
    U, S, Vt = ApproximateRandomizedSVD.randomized_svd(A, rank=10)
    A_approx = U @ np.diag(S) @ Vt
    error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    print(f'  Matrix shape: {A.shape}')
    print(f'  Target rank: 10')
    print(f'  U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}')
    print(f'  Reconstruction error: {error:.4f}')
    print('\n[TEST 2] Compress Linear layer...')
    linear = nn.Linear(512, 256)
    compressor = ModelCompressor(rank_ratio=0.5)
    compressed_linear = compressor.compress_linear(linear, 'test_linear')
    x = torch.randn(4, 512)
    with torch.no_grad():
        y_orig = linear(x)
        y_comp = compressed_linear(x)
    error = torch.norm(y_orig - y_comp) / torch.norm(y_orig)
    print(f'  Original params: {linear.weight.numel()}')
    print(f'  Compressed rank: {compressed_linear.rank}')
    print(f'  Forward pass error: {error.item():.4f}')
    print('\n[TEST 3] Compress Conv1d layer...')
    conv = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    compressed_conv = compressor.compress_conv1d(conv, 'test_conv')
    x = torch.randn(4, 32, 100)
    with torch.no_grad():
        y_orig = conv(x)
        y_comp = compressed_conv(x)
    error = torch.norm(y_orig - y_comp) / torch.norm(y_orig)
    print(f'  Original shape: {conv.weight.shape}')
    print(f'  Compressed rank: {compressed_conv.rank}')
    print(f'  Forward pass error: {error.item():.4f}')
    print('\n[TEST 4] Compress NeuralGraph model...')
    try:
        from model import NeuralGraph, ModelConfig
        config = ModelConfig()
        model = NeuralGraph(config)
        original_params = count_parameters(model)
        original_size = get_model_size_mb(model)
        compressed_model, stats = compress_model(model, rank_ratio=0.5, exclude_layers=['detection_head', 'forecast_head'])
        compressed_params = count_parameters(compressed_model)
        compressed_size = get_model_size_mb(compressed_model)
        print(f'  Original parameters: {original_params:,}')
        print(f'  Compressed parameters: {compressed_params:,}')
        print(f'  Compression ratio: {original_params / compressed_params:.2f}x')
        print(f'  Original size: {original_size:.2f} MB')
        print(f'  Compressed size: {compressed_size:.2f} MB')
        test_input = torch.randn(2, config.NUM_CHANNELS, config.TIME_STEPS)
        error_metrics = evaluate_approximation_error(model, compressed_model, test_input)
        print(f'\n  Approximation Quality:')
        print(f"    MSE: {error_metrics['mse']:.6f}")
        print(f"    Cosine Similarity: {error_metrics['cosine_similarity']:.4f}")
    except ImportError:
        print('  Skipping model compression test (model.py not found)')
    print('\n' + '=' * 60)
    print('ARSVD Compression test completed successfully!')
    print('=' * 60)