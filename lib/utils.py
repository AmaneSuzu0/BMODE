import torch
import numpy as np


def pad_sequence(sequences, padding_value=0):
    """动态填充变长时间序列数据"""
    num_nodes = sequences[0].shape[0]

    # 获取batch内最大时间步长
    max_patch = max(seq.shape[1] for seq in sequences)
    max_len = max(seq.shape[2] for seq in sequences)


    padded_seqs = []
    for seq in sequences:
        pad_patch = max_patch - seq.shape[1]
        pad_len = max_len - seq.shape[2]

        # 四维数据填充 [N, P, T, D]
        if len(seq.shape) == 4:
            padded = np.pad(seq,
                            ((0, 0), (0, pad_patch), (0, pad_len), (0, 0)),
                            mode='constant',
                            constant_values=padding_value)
        # 三维填充[N, P, T]
        elif len(seq.shape) == 3:
            padded = np.pad(seq,
                            ((0, 0), (0, pad_patch), (0, pad_len)),
                            mode='constant',
                            constant_values=0)

        padded_seqs.append(padded)

    return np.stack(padded_seqs)


def pad_timestamp(sequences, padding_value=0):
    """动态填充变长时间序列数据"""
    # 获取batch内最大时间步长
    max_len = max(seq.shape[0] for seq in sequences)

    padded_seqs = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        padded = np.pad(seq,
                        ((0, pad_len), (0, 0)),
                        mode='constant',
                        constant_values=padding_value)
        padded_seqs.append(padded)

    return np.stack(padded_seqs)


class StandardScaler:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

