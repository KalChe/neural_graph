import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, Optional, Tuple
import numpy as np
class SimpleGraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int=64, out_dim: int=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x
class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels: int=22, out_dim: int=64):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(128, out_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return x
class NeuralGraphSimple(nn.Module):
    def __init__(self, num_channels: int=22, time_steps: int=1675, hidden_dim: int=128, use_graph: bool=True, num_classes: int=2):
        super().__init__()
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.use_graph = use_graph
        self.cnn_encoder = SimpleCNNEncoder(num_channels, out_dim=hidden_dim)
        if use_graph:
            self.graph_encoder = SimpleGraphEncoder(in_channels=time_steps, hidden_dim=64, out_dim=hidden_dim // 2)
            classifier_input = hidden_dim + hidden_dim // 2
        else:
            self.graph_encoder = None
            classifier_input = hidden_dim
        self.classifier = nn.Sequential(nn.Linear(classifier_input, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.4), nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim // 2, num_classes))
        self.register_buffer('edge_index', None)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def compute_static_adjacency(self, data_sample: torch.Tensor, top_k: int=6) -> torch.Tensor:
        device = data_sample.device
        batch_size, n_ch, n_time = data_sample.shape
        mean = data_sample.mean(dim=-1, keepdim=True)
        std = data_sample.std(dim=-1, keepdim=True) + 1e-08
        x_norm = (data_sample - mean) / std
        corr = torch.bmm(x_norm, x_norm.transpose(1, 2)) / n_time
        avg_corr = corr.abs().mean(dim=0)
        avg_corr.fill_diagonal_(0)
        _, top_k_idx = torch.topk(avg_corr, k=top_k, dim=-1)
        src = torch.arange(n_ch, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        dst = top_k_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        self.edge_index = edge_index
        return edge_index
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        cnn_features = self.cnn_encoder(x)
        if self.use_graph and (edge_index is not None or self.edge_index is not None):
            if edge_index is None:
                edge_index = self.edge_index
            x_graph = x.transpose(1, 2).reshape(-1, self.time_steps)
            x_graph = x.reshape(batch_size * self.num_channels, self.time_steps)
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.num_channels).reshape(-1)
            edge_list = []
            for i in range(batch_size):
                offset = i * self.num_channels
                edge_list.append(edge_index + offset)
            batched_edge_index = torch.cat(edge_list, dim=1)
            graph_features = self.graph_encoder(x_graph, batched_edge_index, batch_idx)
            combined = torch.cat([cnn_features, graph_features], dim=1)
        else:
            combined = cnn_features
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)
        return {'logits': logits, 'probs': probs, 'detection_logits': logits, 'detection_probs': probs, 'forecast_logits': logits, 'forecast_probs': probs, 'features': combined}
class FocalLoss(nn.Module):
    def __init__(self, alpha: float=0.75, gamma: float=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
def count_parameters(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum((p.numel() * p.element_size() for p in model.parameters()))
    buffer_size = sum((b.numel() * b.element_size() for b in model.buffers()))
    return (param_size + buffer_size) / (1024 * 1024)
if __name__ == '__main__':
    model = NeuralGraphSimple(num_channels=22, time_steps=1675, use_graph=False)
    print(f'Parameters: {count_parameters(model):,}')
    print(f'Size: {get_model_size_mb(model):.2f} MB')
    x = torch.randn(4, 22, 1675)
    out = model(x)
    print(f"Output shape: {out['logits'].shape}")