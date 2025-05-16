# data/dataloader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class StockDataset(Dataset):
    """时间序列数据集生成器"""
    def __init__(self, data, targets, seq_length=10):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class DataProcessor:
    """数据处理管道"""
    def __init__(self, data_dir='/media/disk/02drive/12zesheng/SNN-Net/data/raw/', seq_length=10):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.scalers = {}
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

    def _load_raw_data(self):
        """加载原始CSV数据"""
        df = pd.read_csv(os.path.join(self.data_dir, 'szzs.csv'), 
                        parse_dates=['date'], index_col='date')
        df.sort_index(inplace=True)  # 确保时间顺序
        return df

    def _preprocess_data(self, df):
        """数据预处理流程"""
        # 处理缺失值（前向填充）
        df.ffill(inplace=True)
        
        # 对volume进行对数标准化
        # df['volume'] = np.log1p(df['volume'])
        
        # 分离特征和目标（预测收盘价）
        features = df[['open','high','low','close','adj_close','volume']]
        targets = df['close'].values
        
        return features.values, targets

    def _train_test_split(self, data, test_size=0.3):
        """时序数据分割"""
        n = len(data)
        train_end = int(n * (1 - test_size))
        train_data = data[:train_end]
        test_data = data[train_end:]
        return train_data, test_data

    def _create_sequences(self, data, targets):
        """创建时间序列样本"""
        sequences = []
        labels = []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i+self.seq_length])
            labels.append(targets[i+self.seq_length])
        return np.array(sequences), np.array(labels)

    def _save_data(self, data_dict):
        """保存处理后的数据"""
        for name, data in data_dict.items():
            torch.save(data, os.path.join(self.data_dir, 'processed', f'{name}.pt'))

    def process(self):
        """完整处理流程"""
        # 1. 加载原始数据
        df = self._load_raw_data()
        
        # 2. 预处理
        features, targets = self._preprocess_data(df)
        
        # 3. 划分数据集 (70%训练，15%验证，15%测试)
        train_data, temp_data = self._train_test_split(features, test_size=0.3)
        train_targets, temp_targets = self._train_test_split(targets, test_size=0.3)
        val_data, test_data = self._train_test_split(temp_data, test_size=0.5)
        val_targets, test_targets = self._train_test_split(temp_targets, test_size=0.5)
        
        # 4. 标准化（仅在训练集上拟合）
        self.scalers = {
            'mean': train_data.mean(axis=0),
            'std': train_data.std(axis=0)
        }
        train_data = (train_data - self.scalers['mean']) / self.scalers['std']
        val_data = (val_data - self.scalers['mean']) / self.scalers['std']
        test_data = (test_data - self.scalers['mean']) / self.scalers['std']
        
        # 5. 创建序列样本
        X_train, y_train = self._create_sequences(train_data, train_targets)
        X_val, y_val = self._create_sequences(val_data, val_targets)
        X_test, y_test = self._create_sequences(test_data, test_targets)
        
        # 6. 保存数据
        self._save_data({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scalers': self.scalers
        })
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_dataloaders(self, batch_size=32):
        """获取数据加载器"""
        # 加载已处理数据
        X_train = torch.load(os.path.join(self.data_dir, 'processed/X_train.pt'))
        y_train = torch.load(os.path.join(self.data_dir, 'processed/y_train.pt'))
        X_val = torch.load(os.path.join(self.data_dir, 'processed/X_val.pt'))
        y_val = torch.load(os.path.join(self.data_dir, 'processed/y_val.pt'))
        X_test = torch.load(os.path.join(self.data_dir, 'processed/X_test.pt'))
        y_test = torch.load(os.path.join(self.data_dir, 'processed/y_test.pt'))
        
        # 创建Dataset
        train_dataset = StockDataset(X_train, y_train, self.seq_length)
        val_dataset = StockDataset(X_val, y_val, self.seq_length)
        test_dataset = StockDataset(X_test, y_test, self.seq_length)
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return {'train':train_loader, 'val':val_loader, 'test':test_loader}

if __name__ == "__main__":
    # 示例用法
    processor = DataProcessor(seq_length=14)
    processor.process()
    train_loader= processor.get_dataloaders(batch_size=64)
    
    # 验证数据形状
    sample_x, sample_y = next(iter(train_loader['train']))
    print(f"输入数据形状: {sample_x.shape}")  # 应为 [batch_size, seq_len, features]
    print(f"目标数据形状: {sample_y.shape}")   # 应为 [batch_size, 1]