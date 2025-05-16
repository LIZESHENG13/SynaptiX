# src/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import seaborn as sns  

class Evaluator:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.device = model.device
        
        # 存储结果
        self.true_values = []
        self.predictions = []
    def evaluate(self):
        """执行完整评估流程"""
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                preds, _ = self.model(inputs)
                
                # 关键修改：取最后一个时间窗口预测
                final_preds = preds[:, -1, :]  # 形状变为 [batch_size, 1]
                
                # 存储结果
                self.true_values.append(targets.cpu().numpy().flatten())  # 展平为一维
                self.predictions.append(final_preds.cpu().numpy().flatten())

    def calculate_metrics(self):
        """计算评估指标"""
        y_true = np.concatenate(self.true_values)  # 一维数组
        y_pred = np.concatenate(self.predictions)  # 一维数组
        
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        # y_true = np.concatenate(self.true_values)
        # y_pred = np.concatenate(self.predictions)
        
        # # 将概率转换为类别（假设使用sigmoid输出）
        # y_pred_class = (y_pred > 0.5).astype(int)
        
        # metrics = {
        #     'Accuracy': accuracy_score(y_true, y_pred_class),
        #     'Precision': precision_score(y_true, y_pred_class),
        #     'Recall': recall_score(y_true, y_pred_class),
        #     'F1': f1_score(y_true, y_pred_class),
        #     'AUC': roc_auc_score(y_true, y_pred),
        #     'Confusion_Matrix': confusion_matrix(y_true, y_pred_class)
        # }
        
        # # 添加分类报告
        # print("\nClassification Report:")
        # print(classification_report(y_true, y_pred_class))
        
        # return metrics

    def visualize_results(self, sample_size=300):
        """可视化分析"""
        # 确保先执行评估
        if not self.true_values:
            self.evaluate()
            
        y_true = np.concatenate(self.true_values)
        y_pred = np.concatenate(self.predictions)
        
        # 绘制整体对比图
        plt.figure(figsize=(15, 10))
        
        # 子图1：预测与真实值对比
        plt.subplot(2, 1, 1)
        plt.plot(y_true[:sample_size], label='True', marker='o', alpha=0.7)
        plt.plot(y_pred[:sample_size], label='Pred', marker='x', linestyle='--')
        plt.title(f"Prediction Comparison (First {sample_size} Samples)")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        
        # 子图2：误差分布分析
        plt.subplot(2, 1, 2)
        errors = y_true - y_pred
        plt.hist(errors, bins=30, density=True, alpha=0.7)
        plt.title("Prediction Error Distribution\n" +
                 f"μ={errors.mean():.4f}, σ={errors.std():.4f}")
        plt.xlabel("Prediction Error")
        plt.ylabel("Density")
        
        plt.tight_layout()
        plt.savefig('/media/disk/02drive/12zesheng/SNN-Net/experiments/results/figures/figure1.png')

        # 误差时间分布
        plt.figure(figsize=(12, 4))
        plt.plot(errors[:500], color='red', alpha=0.5)
        plt.title("Error Distribution Over Time (First 500 Samples)")
        plt.xlabel("Time Steps")
        plt.ylabel("Error Magnitude")
        plt.axhline(0, color='black', linestyle='--')
        plt.savefig('/media/disk/02drive/12zesheng/SNN-Net/experiments/results/figures/figure2.png')
        # y_true = np.concatenate(self.true_values)
        # y_pred = np.concatenate(self.predictions)
        
        # # 将概率转换为类别（假设使用sigmoid输出）
        # y_pred_class = (y_pred > 0.5).astype(int)

        # plt.figure(figsize=(15, 10))
        
        # # ROC曲线
        # fpr, tpr, _ = roc_curve(y_true, y_pred)
        # plt.subplot(2, 2, 1)
        # plt.plot(fpr, tpr)
        # plt.title("ROC Curve")
        
        # # 混淆矩阵热力图
        # cm = confusion_matrix(y_true, y_pred_class)
        # plt.subplot(2, 2, 2)
        # sns.heatmap(cm, annot=True, fmt='d')
        # plt.title("Confusion Matrix")

        # plt.savefig('/media/disk/02drive/12zesheng/SNN-Net/experiments/results/figures/figure3.png')

    def error_analysis(self, threshold=0.1):
        """预测误差的深入分析"""
        errors = np.abs(np.concatenate(self.true_values) - np.concatenate(self.predictions))
        error_rate = np.mean(errors > threshold)
        
        print(f"Error Analysis Report:")
        print(f"- 平均绝对误差: {np.mean(errors):.4f}")
        print(f"- 最大误差值: {np.max(errors):.4f}")
        print(f"- 误差超过{threshold*100}%的比例: {error_rate*100:.2f}%")
        
        # 识别前5%的最大误差样本
        large_error_indices = np.argsort(errors)[-len(errors)//20:]
        print("\n典型误差案例特征分析:")
        print(f"最大值分布区间: [{np.percentile(errors, 95):.4f}, {np.max(errors):.4f}]")