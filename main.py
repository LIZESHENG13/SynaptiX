# main.py
import os
import yaml
import torch
from models.full_model import SNNMemoryModel
from data.dataloader import DataProcessor
from train import Trainer
from evaluate import Evaluator
import warnings
# from src.utils import setup_logger, get_device
warnings.filterwarnings('ignore') 


def main():
    # 初始化系统设置
    device = "cuda" if torch.cuda.is_available() else "cpu" # 自动检测GPU
    
    # 加载配置
    with open('/media/disk/02drive/12zesheng/SNN-Net/configs/train_config.yaml') as f:
        config = yaml.safe_load(f)

    # 数据准备
    data_processor = DataProcessor(
        data_dir=config['data']['raw_dir'],  
        seq_length=config['training']['seq_length'] 
    )
    
    try:
        # 尝试加载已处理数据
        train_loader = data_processor.get_dataloaders(
            batch_size=config['training']['batch_size']
        )
    except FileNotFoundError:
        # 数据预处理流程
        print("Starting data preprocessing...")
        data_processor.process()
        train_loader = data_processor.get_dataloaders(
            batch_size=config['training']['batch_size']
        )

    # 模型初始化
    model = SNNMemoryModel(config['model']).to(device)
    print(f"Model architecture:\n{model}")

    # 训练流程
    trainer = Trainer(
        model=model,
        config=config['training'],
        dataloaders=train_loader
    )
    
    print("Starting training process...")
    trainer.train()
    print("Model saved to /media/disk/02drive/12zesheng/SNN-Net/experiments/results/checkpoints/组合5/snn_memory_model.pth")
    
    # 保存最终模型
    checkpoint = torch.load('/media/disk/02drive/12zesheng/SNN-Net/experiments/results/checkpoints/组合5/snn_memory_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试集评估
    print("Starting final evaluation...")
    evaluator = Evaluator(model, train_loader['test'])
    evaluator.evaluate()
    
    # 结果分析
    metrics = evaluator.calculate_metrics()
    print(f"Final Test Metrics:\n" , metrics.items())
    
    # 生成可视化报告
    evaluator.visualize_results(sample_size=200)
    evaluator.error_analysis(threshold=0.05)
    print("Visual report generated")

if __name__ == "__main__":
    main()