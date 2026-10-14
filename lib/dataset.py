import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import pad_sequence, pad_timestamp

class SpatioTemporalDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: 原始数据列表，每个元素为[features, obs_mask, pred_mask]
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, obs_mask, pred_mask = self.data[idx]
        # 时间戳列（第0列）保持 float64，其他列转换为 float32
        features_processed = features.copy()
        features_processed[..., 1:] = features[..., 1:].astype(np.float32)  # 第1列及以后转为 float32
        # 第0列（时间戳）保持原始类型（应该是 float64）
        
        return {
            'features': features_processed,  # 时间戳保持 float64，其他为 float32
            'obs_mask': obs_mask.astype(np.float32),
            'pred_mask': pred_mask.astype(np.float32),
        }


def collate_fn(batch):
    """自定义批处理函数"""
    features = [item['features'] for item in batch]
    obs_masks = [item['obs_mask'] for item in batch]
    pred_masks = [item['pred_mask'] for item in batch]

    # 动态填充
    padded_features = pad_sequence(features)
    padded_obs_masks = pad_sequence(obs_masks)
    padded_pred_masks = pad_sequence(pred_masks)

    # 转换为PyTorch张量
    features_tensor = torch.tensor(padded_features)
    obs_masks_tensor = torch.tensor(padded_obs_masks)
    pred_masks_tensor = torch.tensor(padded_pred_masks)

    return {
        'features': features_tensor,
        'obs_mask': obs_masks_tensor,
        'pred_mask': pred_masks_tensor,
    }


def dataset_split(data_list, train_ratio=0.6, val_ratio=0.2):
    """
    划分数据集
    """
    total_samples = len(data_list)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]

    return train_data, val_data, test_data


def create_dataloaders(data_list, cfg):
    """创建训练、验证、测试数据加载器"""
    # 划分数据集
    train_data, val_data, test_data = dataset_split(
        data_list,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO
    )

    print(f"数据集大小: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")

    # 创建数据集实例
    train_dataset = SpatioTemporalDataset(train_data)
    val_dataset = SpatioTemporalDataset(val_data)
    test_dataset = SpatioTemporalDataset(test_data)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,  #
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,  # ️
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,  #
    )

    return train_loader, val_loader, test_loader
