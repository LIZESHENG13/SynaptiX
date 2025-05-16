# models/reducer.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from joblib import dump, load

class Autoencoder(nn.Module):
    """增强型自编码器"""
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*4, latent_dim*2),
            nn.Tanh(),
            nn.Linear(latent_dim*2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim*2, latent_dim*4),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class DimensionalityReducer(nn.Module):
    """支持多种降维方法的统一接口"""
    def __init__(self, method='autoencoder', input_dim=128, latent_dim=32, device='auto'):
        super().__init__()
        self.is_fitted = False
        self.method = method
        self.latent_dim = latent_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device=='auto' else device
        ) if isinstance(device, str) else device
        
        # 初始化降维模型
        if method == 'autoencoder':
            self.model = Autoencoder(input_dim, latent_dim).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            self.criterion = nn.MSELoss()
        elif method == 'pca':
            self.model = PCA(n_components=latent_dim)
        elif method == 'umap':
            self.model = UMAP(n_components=latent_dim, random_state=42)
        elif method == 't-sne':
            self.model = TSNE(n_components=latent_dim,
                method='barnes_hut' if self.latent_dim <= 3 else 'exact')
        else:
            raise ValueError(f"不支持的降维方法: {method}")

    def forward(self, x):
        """处理时序数据的降维"""
        # 输入形状: [batch*time_win, seq_len, features]
        orig_device = x.device
        batch, seq_len, feat_dim = x.shape
        
        # 自动设备转移
        if self.method == 'autoencoder':
            x = x.to(self.device)
        
        # 批量处理所有时间步
        x_flat = x.view(-1, feat_dim)
        reduced = self._reduce_batch(x_flat)
        
        # 恢复原始形状和设备
        return reduced.view(batch, seq_len, self.latent_dim).to(orig_device)
    
    def fit(self, X):
        """统一训练/拟合接口"""
        if self.method != 'autoencoder':
            # 传统方法拟合
            X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.array(X)
            if self.method == 't-sne':
                print(X_np)
                print(X_np.shape)
                self.model.fit(X_np)  # t-SNE不需要transform
            else:
                self.model.fit(X_np)
        
        self.is_fitted = True
    
    def _reduce_batch(self, X):
        """核心降维方法"""
        if not self.is_fitted:
            self.fit(X)
        if self.method == 'autoencoder':
            self.model.eval()
            with torch.no_grad():
                encoded, _ = self.model(X.to(self.device))
            return encoded.cpu()
            
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.array(X)
        
        if self.method == 't-sne':
            return torch.from_numpy(self.model.fit_transform(X_np)).float()
            
        if self.method == 'umap':
            return torch.from_numpy(self.model.transform(X_np)).float()
            
        return torch.from_numpy(self.model.transform(X_np)).float()

    def evaluate(self, X):
        """跨方法的降维质量评估"""
        if self.method == 'autoencoder':
            with torch.no_grad():
                _, reconstructed = self.model(X.to(self.device))
            return self.criterion(reconstructed, X).item()
        elif self.method == 'pca':
            return np.sum(self.model.explained_variance_ratio_)
        elif self.method == 'umap':
            emb = self.model.transform(X.cpu().numpy())
            return (emb.max(axis=0) - emb.min(axis=0)).mean()  # 跨度指标
        elif self.method == 't-sne':
            return 0.0  # t-SNE无明确评估指标
        return 0.0

    def save(self, path):
        """统一保存接口"""
        if self.method == 'autoencoder':
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        else:
            dump(self.model, path)

    def load(self, path):
        """统一加载接口"""
        if self.method == 'autoencoder':
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.model = load(path)

    def to(self, device):
        """设备转移增强"""
        if self.method == 'autoencoder':
            super().to(device)
            self.device = device
        return self

# 使用示例
if __name__ == "__main__":
    # 初始化
    reducer = DimensionalityReducer(method='t-sne', input_dim=128, latent_dim=5)
    
    # 生成模拟数据
    dummy_data = torch.randn(1000, 128)
    
    # 训练/拟合
    reducer.fit(dummy_data)
    
    # 执行降维
    reduced_data = reducer._reduce_batch(dummy_data[:40])
    print(f"降维结果尺寸：{reduced_data.shape}")
    
    # 质量评估
    quality = reducer.evaluate(dummy_data)
    print(f"降维质量指标：{quality:.4f}")
    
    # 保存模型
    # reducer.save_model("dim_reducer.pkl")