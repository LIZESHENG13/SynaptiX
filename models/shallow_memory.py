# models/shallow_memory.py
import torch
import torch.nn as nn
from snntorch import surrogate

class ShallowBase(nn.Module):
    """浅层记忆基类（含通用功能）"""
    def __init__(self, freeze_threshold=0.2):
        super().__init__()
        self.freeze_threshold = freeze_threshold
        self.register_buffer('_velocity_buffer', None)
        
        # STDP基础参数
        self.learning_rate = 0.01
        self.register_buffer('tau_pre', torch.tensor(20e-3)) 
        self.register_buffer('tau_post', torch.tensor(20e-3))  
        self.register_buffer('prev_pre_spikes', None)
        self.register_buffer('prev_post_spikes', None)

    def compute_velocity(self, x):
        """带缓冲区管理的速度计算"""
        if self._velocity_buffer is None or self._velocity_buffer.shape[0] != x.shape[0]:
            self._velocity_buffer = x.detach().clone()
            return torch.zeros_like(x)
        velocity = x - self._velocity_buffer
        self._velocity_buffer.data.copy_(x.detach())  # 保持梯度分离
        return velocity

    def apply_freeze(self, activation):
        """改进的冻结机制，保留梯度流"""
        activation_norm = torch.norm(activation, dim=-1, keepdim=True)
        mask = (activation_norm > self.freeze_threshold).float()
        return activation * mask + (1 - mask) * activation.detach()
        # return activation

    def stdp_update(self, pre_spike, post_spike):
        """STDP通用更新规则"""
        if (self.prev_pre_spikes is None or 
            self.prev_pre_spikes.size(0) != pre_spike.size(0)):
            self.prev_pre_spikes = pre_spike.detach()
            self.prev_post_spikes = post_spike.detach()
            return
        
        delta_t = post_spike.unsqueeze(-1) - pre_spike.unsqueeze(1)
        
        ltp = torch.where(delta_t > 0, 
                         torch.exp(-delta_t / self.tau_pre.clamp(min=1e-6)),
                         torch.zeros_like(delta_t))
        ltd = torch.where(delta_t < 0,
                         -torch.exp(delta_t / self.tau_post.clamp(min=1e-6)),
                         torch.zeros_like(delta_t))
        
        delta_w = self.learning_rate * (ltp + ltd).mean(dim=0)
        if hasattr(self, 'fc'):  # 适用于全连接层
            self.fc.weight.data.add_(delta_w)
        elif hasattr(self, 'conv'):  # 适用于卷积层
            self.conv[0].weight.data.add_(delta_w.mean(dim=-1))
        
        # 更新历史记录
        self.prev_pre_spikes = 0.9*self.prev_pre_spikes + 0.1*pre_spike.detach()
        self.prev_post_spikes = 0.9*self.prev_post_spikes + 0.1*post_spike.detach()
    
    def generate_feedback(self):
        """基于存储记忆生成反馈信号"""
        if self.memory_buffer is None:
            return torch.zeros((1, self.input_dim), device=self.feedback_layer[0].weight.device)
        
        # 通过基类的反馈层处理记忆 [batch, memory_dim] → [batch, input_dim]
        return self.feedback_layer(self.memory_buffer)
    
class ShallowSNN(ShallowBase):
    """脉冲神经网络浅层（完整特性版）"""
    def __init__(self, input_dim, hidden_dim, freeze_threshold=0.2):
        super().__init__(freeze_threshold)
        self.fc = nn.Linear(input_dim * 2, hidden_dim)
        self.spike_grad = surrogate.fast_sigmoid()
        
        # SNN特有参数
        self.tau_mem = nn.Parameter(torch.tensor(10.0))  # 膜电位时间常数

    def forward(self, x):
        # 特征增强
        velocity = self.compute_velocity(x)
        x_aug = torch.cat([x, velocity], dim=-1)
        
        # 脉冲生成
        mem_potential = self.fc(x_aug)
        spikes = torch.sigmoid(mem_potential / self.tau_mem)
        
        # 动态冻结
        filtered_spikes = self.apply_freeze(spikes)
        
        # STDP更新（仅训练模式）
        if self.training:
            self.stdp_update(x_aug.detach(), filtered_spikes.detach())
        
        return filtered_spikes

class ShallowRNN(ShallowBase):
    """RNN浅层（带特性增强）"""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm', freeze_threshold=0.2):
        super().__init__(freeze_threshold)
        self.rnn = getattr(nn, rnn_type.upper())(
            input_dim + input_dim,  # 增加速度特征维度
            hidden_dim,
            batch_first=True
        )
    
    def forward(self, x):
        # 速度特征融合
        velocity = self.compute_velocity(x)
        x_aug = torch.cat([x, velocity], dim=-1)
        
        # RNN处理
        output, _ = self.rnn(x_aug)
        
        # 动态冻结
        return self.apply_freeze(output)

class ShallowCNN(ShallowBase):
    """CNN浅层（带时空特性）修正版"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, freeze_threshold=0.2):
        super().__init__(freeze_threshold)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1,         # 关键修改1：输入通道设为1
                     out_channels=hidden_dim, 
                     kernel_size=kernel_size,
                     padding=kernel_size//2),
            nn.ReLU()
        )
    
    def forward(self, x):
        """前向传播流程
        输入形状: [batch*time_win, features]
        输出形状: [batch*time_win, hidden_dim]
        """
        # 速度特征融合
        velocity = self.compute_velocity(x)
        x_aug = torch.cat([x, velocity], dim=1)  # 形状 [N, 2*features]
        
        # 添加通道维度
        x_aug = x_aug.unsqueeze(1)           # 形状 [N, 1, 2*features]
        
        # CNN处理
        features = self.conv(x_aug)          # 输出 [N, hidden_dim, 2*features]
        
        # 特征聚合
        features = features.mean(dim=-1)     # 全局平均池化 [N, hidden_dim]
        
        return self.apply_freeze(features)
    
class ShallowTransformer(ShallowBase):
    """Transformer浅层（带时空注意力）"""
    def __init__(self, input_dim, hidden_dim, num_heads=4, freeze_threshold=0.2):
        super().__init__(freeze_threshold)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim*2,  # 包含速度特征
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=2
        )
        self.proj = nn.Linear(input_dim*2, hidden_dim)
    
    def forward(self, x):
        # 速度特征融合
        velocity = self.compute_velocity(x)
        x_aug = torch.cat([x, velocity], dim=-1)
        # Transformer处理
        x_aug = x_aug.unsqueeze(0)
        encoded = self.encoder(x_aug)
        projected = self.proj(encoded.squeeze(0))
        # 动态冻结
        return self.apply_freeze(projected)
        