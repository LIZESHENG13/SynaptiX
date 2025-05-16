# models/deep_memory.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DeepMemoryBase(nn.Module):
    """深度记忆基类"""
    def __init__(self, input_dim, memory_dim):
        super().__init__()
        self.memory_dim = memory_dim
        self.input_dim = input_dim
        self.register_buffer('memory_buffer', None)
        
        # 通用反馈机制
        self.feedback_layer = nn.Sequential(
            nn.Linear(memory_dim, input_dim),
            nn.Sigmoid()
        )

    def generate_feedback(self):
        """生成反馈信号（需子类实现）"""
        raise NotImplementedError

    def build_memory_graph(self, x):
        """构建时空图结构（用于GNN）"""
        # 默认实现：时间邻接
        batch, seq_len, _ = x.shape
        edges = []
        for i in range(seq_len-1):
            edges.append([i, i+1])
            edges.append([i+1, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x):
        """前向传播（需子类实现）"""
        raise NotImplementedError

class TransformerMemory(DeepMemoryBase):
    """组合1/4：Transformer架构"""
    def __init__(self, input_dim, memory_dim, num_layers=4, nhead=8):
        super().__init__(input_dim, memory_dim)
        # 添加维度投影层
        self.input_proj = nn.Linear(input_dim, memory_dim)
        self.pos_encoder = PositionalEncoding(memory_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=memory_dim,
            nhead=nhead,
            dim_feedforward=4*memory_dim
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: [batch, seq, features]
        x = x.permute(1, 0, 2)  # [seq, batch, features]
        x = self.input_proj(x)
        memory = self.pos_encoder(x)
        
        # 自注意力记忆更新
        tgt = torch.zeros_like(x)
        memory = self.decoder(tgt, memory)
        
        # 存储记忆
        self.memory_buffer = memory.detach()
        return memory[-1], self.generate_feedback()
    
    def generate_feedback(self):
        """基于注意力权重的反馈信号"""
        if self.memory_buffer is None:
            return super().generate_feedback()
            
        # 使用第一层注意力权重
        memory = self.memory_buffer.permute(1, 0, 2)  # [seq, batch, feat] -> [batch, seq, feat]
        attn_output, attn_weights = self.decoder.layers[0].self_attn(
            memory, memory, memory,
            need_weights=True
        )
        return attn_weights.mean(dim=1)  # [batch, seq, seq] -> [batch, seq]

class GNNDynamicMemory(DeepMemoryBase):
    """修正后的GNN记忆模块（带有效边生成）"""
    def __init__(self, input_dim, memory_dim):
        super().__init__(input_dim, memory_dim)
        
        # GNN层维度验证
        # assert input_dim == memory_dim, "输入维度必须等于memory_dim"
        self.conv1 = GCNConv(input_dim, memory_dim)
        self.conv2 = GCNConv(memory_dim, memory_dim)
        
        # 边生成器改进
        self.edge_generator = nn.Sequential(
            nn.Linear(input_dim*2, 64),  # 考虑节点对特征
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def build_memory_graph(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 生成候选边
        node_indices = torch.arange(batch_size * seq_len, device=x.device)
        # 每个节点连接5个邻居，确保形状兼容
        src = node_indices.view(-1, 1).repeat(1, 5)  # 形状: (batch*seq_len, 5)
        # 生成随机目标索引，确保形状为 (batch*seq_len, 5)
        dst = (node_indices.view(-1, 1) + \
                    torch.randint(1, seq_len, size=(node_indices.size(0), 5), device=x.device)) % seq_len
        # 展平以匹配维度
        candidate_edges = torch.stack([src.flatten(), dst.flatten()], dim=0)
        
        # 计算边权重
        node_pairs = torch.cat([
            x.reshape(-1, self.input_dim)[candidate_edges[0]], 
            x.reshape(-1, self.input_dim)[candidate_edges[1]]
        ], dim=1)
        edge_weights = torch.sigmoid(self.edge_generator(node_pairs)).flatten()
        
        # 选择top-k有效边
        k = int(candidate_edges.size(1) * 0.3)  # 保留30%的边
        topk_indices = torch.topk(edge_weights, k=k).indices
        return candidate_edges[:, topk_indices]
    
    def generate_feedback(self, x):
        """基于GNN输出的反馈信号"""
        # x的形状: [batch, seq_len, memory_dim]
        # 聚合时序特征并生成反馈
        aggregated = x.mean(dim=1)  # [batch, memory_dim]
        feedback = self.feedback_layer(aggregated)  # 通过反馈层
        return feedback

    def forward(self, x):
        """
        输入形状: [batch, seq_len, input_dim]
        输出形状: [batch, memory_dim]
        """
        batch, seq_len, _ = x.shape
        
        # 构建图结构
        edge_index = self.build_memory_graph(x)
        
        # 节点特征处理
        nodes = x.reshape(-1, self.input_dim)
        
        # 图卷积操作
        x = F.relu(self.conv1(nodes, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        
        print(x.shape)
        # 聚合特征（按样本）
        return x.view(batch, seq_len, -1).mean(dim=1), self.generate_feedback(x.view(batch, seq_len, -1))
     
class HybridMemory(DeepMemoryBase):
    """组合3/5：混合记忆架构"""
    def __init__(self, input_dim, memory_dim, mode='gru'):
        super().__init__(input_dim, memory_dim)
        self.mode = mode
        
        if mode == 'gru':
            self.rnn = nn.GRU(input_dim, memory_dim, batch_first=True)
        elif mode == 'lstm':
            self.rnn = nn.LSTM(input_dim, memory_dim, batch_first=True)
        elif mode == 'snn':
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, memory_dim*2),
                nn.ReLU(),
                nn.Linear(memory_dim*2, memory_dim)
            )
        
    def forward(self, x):
        if self.mode in ['gru', 'lstm']:
            output, hidden = self.rnn(x)
            self.memory_buffer = hidden.detach()
            return output[:,-1,:], self.generate_feedback()
        else:
            memory = self.mlp(x.mean(dim=1))
            self.memory_buffer = memory.unsqueeze(1)
            return memory, self.generate_feedback()
        
    def generate_feedback(self):
        """基于存储记忆生成反馈信号"""
        if self.memory_buffer is None:
            return torch.zeros((1, self.input_dim), device=self.feedback_layer[0].weight.device)
        
        # 通过基类的反馈层处理记忆 [batch, memory_dim] → [batch, input_dim]
        # return self.feedback_layer(self.memory_buffer)
        return self.feedback_layer(self.memory_buffer)
        # return self.memory_buffer
    
class HierarchicalMemory(DeepMemoryBase):
    """组合4：层次化记忆"""
    def __init__(self, input_dim, hidden_dim):  # 修改参数名以明确含义
        # 基类memory_dim设为2*hidden_dim
        # print(hidden_dim)
        super().__init__(input_dim, memory_dim=2*hidden_dim)  
        
        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=4*input_dim
            ),
            num_layers=2
        )
        # 双向LSTM，每个方向隐藏层维度为hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        
    def forward(self, x):
        # Transformer编码
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        encoded = self.encoder(x)
        
        # LSTM记忆存储
        _, (hidden, _) = self.lstm(encoded)
        self.memory_buffer = hidden.detach()  # 存储双向隐藏状态
        
        # 拼接双向输出作为最终记忆
        combined = torch.cat([hidden[0], hidden[1]], dim=-1)
        print('combined',combined.shape)
        return combined, self.generate_feedback()
    
    def generate_feedback(self):
        """基于LSTM隐藏状态的反馈生成"""
        if self.memory_buffer is None:
            return torch.zeros((1, self.input_dim), device=self.feedback_layer[0].weight.device)
        
        # 获取存储的双向隐藏状态 
        hidden = self.memory_buffer
        
        # 合并双向特征：拼接后通过基类的反馈层
        batch_size = hidden.size(1)
        combined = hidden.permute(1, 0, 2).reshape(batch_size, -1)  # [batch, 2*hidden_dim]
        # 通过反馈层
        return self.feedback_layer(combined)

class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 自动匹配不同输入形状
        if x.dim() == 3:  # [batch, seq, feat]
            seq_len = x.size(1)
            pe = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq, feat]
        elif x.dim() == 2:  # [seq, feat]
            seq_len = x.size(0)
            pe = self.pe[:seq_len, :]
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
        return x + pe.to(x.device)
