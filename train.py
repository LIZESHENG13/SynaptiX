# src/train.py
import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Trainer:
    def __init__(self, model, config, dataloaders):
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # 早停机制
        self.early_stopper = EarlyStopper(
            patience=config['patience'],
            min_delta=0.001
        )
        
        # 训练记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.dataloaders['train'], desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.model.device)
            targets = targets.to(self.model.device)
            # 前向传播
            preds, _ = self.model(inputs)

            # 关键修改：取最后一个时间窗口预测
            final_preds = preds[:, -1, :]
            
            # 调整维度匹配
            loss = self.criterion(final_preds, targets.float())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.dataloaders['train'])

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                inputs, targets = batch
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)
                
                preds, _ = self.model(inputs)
                final_preds = preds[:, -1, :]
                
                # 收集数据
                all_preds.append(final_preds.cpu())
                all_targets.append(targets.cpu())
                
                # 计算loss
                loss = self.criterion(final_preds, targets)
                total_loss += loss.item()
        
        # 合并所有batch结果
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_targets = torch.cat(all_targets).numpy().flatten()
        # 计算全局指标
        metrics = self.calculate_metrics(all_targets, all_preds)
        return total_loss / len(self.dataloaders['val']), metrics

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.history['val_loss'][-1],
        }, f"/media/disk/02drive/12zesheng/SNN-Net/experiments/results/checkpoints/组合5/snn_memory_model.pth")

    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 打印完整指标
            metric_str = " | ".join([f"{k}:{v}" for k,v in metrics.items()])
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Valid Metrics: {metric_str}")
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)
            
            # 早停判断
            if self.early_stopper.should_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break
                
            # 保存最佳模型
            if val_loss == min(self.history['val_loss']):
                self.save_checkpoint(epoch)
    
    def calculate_metrics(self, y_true, y_pred):
        """完善指标计算和异常处理"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 添加百分比指标
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            
            # 添加分布统计
            error_stats = {
                'max_error': np.max(np.abs(y_true - y_pred)),
                'std_error': np.std(y_true - y_pred)
            }
            
            return {
                'MSE': round(mse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'MAPE(%)': round(mape, 2),
                'SMAPE(%)': round(smape, 2),
                **error_stats
            }
        except Exception as e:
            print(f"指标计算错误: {str(e)}")
            return {}


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def should_stop(self, val_loss):
        if val_loss < self.min_loss - self.min_delta:
            self.min_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience