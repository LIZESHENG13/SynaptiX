# models/full_model.py
import torch
import torch.nn as nn
from .shallow_memory import (ShallowSNN, ShallowRNN, 
                           ShallowCNN, ShallowTransformer)
from .reducer import DimensionalityReducer
from .deep_memory import (TransformerMemory, GNNDynamicMemory,
                        HybridMemory, HierarchicalMemory)
import unittest
import warnings
from torch.testing import assert_close

class SNNMemoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化浅层记忆模块
        self.shallow = self._build_shallow(config)
        
        # 初始化降维模块
        self.reducer = DimensionalityReducer(
            method=config['reduce_method'],
            input_dim=config['shallow_hidden'],
            latent_dim=config['latent_dim'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        

        # 初始化深层记忆模块
        self.deep = self._build_deep(config)
        
        # 动态构建预测头
        self.predictor = self._build_predictor(config)
        
        # 设备管理
        self.device = torch.device(config.get('device', 
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def _build_shallow(self, config):
        """构建浅层记忆模块"""
        shallow_type = config.get('shallow_type', 'snn')
        params = {
            'input_dim': config['input_dim'],
            'hidden_dim': config['shallow_hidden'],
            'freeze_threshold': config.get('freeze_threshold', 0.2)
        }
        
        builders = {
            'snn': lambda: ShallowSNN(**params),
            'rnn': lambda: ShallowRNN(**params, rnn_type=config.get('rnn_type', 'lstm')),
            'cnn': lambda: ShallowCNN(**params, kernel_size=config.get('kernel_size', 3)),
            'transformer': lambda: ShallowTransformer(
                **params, 
                num_heads=config.get('num_heads', 2)
            )
        }
        return builders[shallow_type]()

    def _build_deep(self, config):
        """构建深层记忆模块"""
        deep_type = config.get('deep_type', 'transformer')
        input_dim = config['latent_dim']
        memory_dim = config['memory_dim']
        
        builders = {
            # 组合1: TransformerDecoder
            'transformer': lambda: TransformerMemory(
                input_dim,
                memory_dim,
                num_layers=config.get('num_layers', 4)),
            # 组合2: GNN
            'gnn': lambda: GNNDynamicMemory(input_dim, memory_dim),
            # 组合3/5: 混合架构
            'hybrid': lambda: HybridMemory(
                input_dim,
                memory_dim,
                mode=config.get('hybrid_mode', 'gru')),
            # 组合4: 层次化记忆
            'hierarchical': lambda: HierarchicalMemory(input_dim, memory_dim)
        }
        return builders[deep_type]()

    def _build_predictor(self, config):
        """动态构建预测头"""
        in_features = config['shallow_hidden'] + config['memory_dim']
        # print(in_features)
        return nn.Sequential(
            nn.Linear(in_features, config.get('pred_hidden', 64)),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(config.get('pred_hidden', 64), config['output_dim'])
        )

    def forward(self, x):
        """
        输入形状: [batch, time_windows, seq_len, features]
        输出形状: [batch, time_windows, output_dim]
        """
        batch, time_win, seq_len, feat = x.shape
        
        # 浅层特征提取
        shallow_outs = []
        for t in range(seq_len):
            # 并行处理所有时间窗口
            x_t = x[:, :, t, :].view(batch*time_win, feat)
            shallow_out = self.shallow(x_t)  # [batch*time_win, hidden]
            shallow_outs.append(shallow_out)
        
        # 组合时序特征 [batch*time_win, seq_len, hidden]
        shallow_seq = torch.stack(shallow_outs, dim=1)
        # 降维处理 [batch*time_win, seq_len, latent]
        compressed = self.reducer(shallow_seq)
        # 深层记忆处理
        if isinstance(self.deep, GNNDynamicMemory):
            # 特殊处理GNN的图结构输入
            compressed = compressed.view(batch, time_win, seq_len, -1)
            memory_states = []
            for t in range(time_win):
                mem, _ = self.deep(compressed[:, t])
                memory_states.append(mem)
            memory_state = torch.stack(memory_states, dim=1)
        else:
            # 标准处理流程
            memory_state, _ = self.deep(compressed)
            memory_state = memory_state.view(batch, time_win, -1)

        # print('memory_state',memory_state.shape)
        # 特征融合
        last_shallow = shallow_seq[:, -1, :].view(batch, time_win, -1)
        # print('last_shallow',last_shallow.shape)
        # combined = torch.cat([last_shallow, memory_state], dim=-1)
        
        combined = torch.cat([last_shallow, memory_state], dim=-1)

        # 最终预测
        # print('combined',combined.shape)
        prediction = self.predictor(combined)
        return prediction, memory_state

    def get_config_template(self):
        """返回配置模板"""
        return {
            # 基础参数
            'input_dim': 64,
            'output_dim': 10,
            'shallow_hidden': 128,
            'latent_dim': 32,
            'memory_dim': 64,
            
            # 模块选择
            'shallow_type': 'snn',  # ['snn', 'rnn', 'cnn', 'transformer']
            'reduce_method': 'pca', # ['pca', 'umap', 't-sne', 'autoencoder']
            'deep_type': 'transformer', # ['transformer', 'gnn', 'hybrid', 'hierarchical']
            
            # 特殊参数
            'freeze_threshold': 0.2,
            'num_heads': 4,
            'kernel_size': 3,
            'hybrid_mode': 'gru',  # ['gru', 'lstm', 'snn']
            'num_layers': 4,
            'pred_hidden': 64,
            'dropout': 0.2,
            'device': 'auto'
        }


test_configs = {
    "base": {
        'input_dim': 6,
        'output_dim': 10,
        'shallow_hidden': 128,
        'latent_dim': 32,
        'memory_dim': 64,
        'device': 'cpu'
    },
    "combinations": {
        "组合1": {
            "shallow_type": "rnn",
            "reduce_method": "umap",
            "deep_type": "transformer",
            "rnn_type": "lstm",
            "num_heads": 8
        },
        "组合2": {
            "shallow_type": "cnn",
            "reduce_method": "pca",
            "deep_type": "gnn",
            "kernel_size": 5
        },
        "组合3": {
            "shallow_type": "transformer",
            "reduce_method": "autoencoder",
            "deep_type": "hierarchical"
        },
        "组合4": {
            "shallow_type": "snn",
            "reduce_method": "pca",
            "deep_type": "hybrid",
            "hybrid_mode": "snn"
        },
        "组合5": {  # 添加缺失的配置
            "shallow_type": "rnn",
            "reduce_method": "t-sne",
            "deep_type": "hybrid",
            "hybrid_mode": "gru"
        },
    }
}

def run_test(config_name, combination):
    """执行单个组合测试"""
    # try:
    print(f"\n{'='*40}\n正在测试: {config_name}\n{'='*40}")
    
    # 合并配置
    full_config = {**test_configs['base'], **combination}
    print("完整配置:", full_config)

    # 初始化模型
    model = SNNMemoryModel(full_config)
    
    # 生成测试数据
    test_data = torch.randn(64, 14, 14, 6)  # [batch, time_windows, seq_len, features]
    print("输入形状:", test_data.shape)

    # 前向传播
    prediction, memory = model(test_data)
    
    # 验证输出形状
    assert prediction.shape == (64, 14, 10), f"预测形状错误: {prediction.shape}"
    assert memory.shape == (64, 14, 64), f"记忆形状错误: {memory.shape}"
    
    print(f"✅ {config_name} 测试通过")
    print("预测头输出示例:", prediction[0, 0, :5].detach().numpy())
    return True
    
    # except Exception as e:
    #     print(f"❌ {config_name} 测试失败")
    #     print("错误信息:", str(e))
    #     return False

def test_edge_cases():
    """边界条件测试"""
    print("\n{'='*40}\n边界条件测试\n{'='*40}")
    # try:
        # 短序列测试
    short_data = torch.randn(2, 1, 1, 6)
    model = SNNMemoryModel(test_configs['base'])
    prediction, _ = model(short_data)
    assert prediction.shape == (2, 1, 10), "短序列形状错误"
    print("✅ 短序列测试通过")

    # 高维测试
    high_dim_config = {**test_configs['base'], 'latent_dim': 128}
    model = SNNMemoryModel(high_dim_config)
    prediction, _ = model(torch.randn(64, 14, 14, 6))
    assert prediction.shape[-1] == 10, "高维输出错误"
    print("✅ 高维测试通过")
        
    # except Exception as e:
    #     print("❌ 边界条件测试失败:", str(e))

def test_config_template():
    """配置模板测试"""
    print("\n{'='*40}\n配置模板测试\n{'='*40}")
    # try:
    template = SNNMemoryModel.get_config_template()
    model = SNNMemoryModel(template)
    test_data = torch.randn(64, 14, 14, template['input_dim'])
    prediction, _ = model(test_data)
    assert prediction.shape == (64, 14, 10), "模板配置输出形状错误"
    print("✅ 配置模板测试通过")
    # except Exception as e:
    #     print("❌ 配置模板测试失败:", str(e))

if __name__ == '__main__':
    # 禁用警告干扰
    warnings.filterwarnings("ignore")
    
    # 执行组合测试
    results = []
    for name, config in test_configs['combinations'].items():
        results.append(run_test(name, config))
    
    # 执行特殊测试
    test_edge_cases()
    test_config_template()
    
    # 统计结果
    print("\n{'='*40}\n测试汇总:")
    print(f"通过测试: {sum(results)}/{len(results)}")
    print(f"成功率: {sum(results)/len(results)*100:.1f}%")   