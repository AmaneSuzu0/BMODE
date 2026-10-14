import matplotlib

matplotlib.use('Agg')  # 强制无图形后端
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import seaborn as sns
import pandas as pd
from datetime import datetime


def plot_attention_heatmap(attn_tensor, attn_tensor_b, save_path=None, layer_idx=None):
    """
    绘制注意力热力图，成对显示
    
    Args:
        attn_tensor: 第一个注意力张量 (nodeA_attn)
        attn_tensor_b: 第二个注意力张量 (nodeB_attn)，并排显示
        save_path: 保存路径
        layer_idx: 层索引，用于标题
    """
    # 设置科研级绘图风格
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    
    # 提取第一个注意力数据
    attn_a = attn_tensor[0].numpy() if isinstance(attn_tensor, torch.Tensor) else attn_tensor[0]
    # attn_a: N, T 或 T, N，需要确认维度
    if attn_a.ndim == 3:
        attn_a = attn_a.mean(axis=0)  # 如果有多头，取平均
    
    # 提取第二个注意力数据
    attn_b = attn_tensor_b[0].numpy() if isinstance(attn_tensor_b, torch.Tensor) else attn_tensor_b[0]
    if attn_b.ndim == 3:
        attn_b = attn_b.mean(axis=0)
    
    # 创建左右并排的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 统一颜色范围
    vmin = min(attn_a.min(), attn_b.min())
    vmax = max(attn_a.max(), attn_b.max())
    
    # ========== 左侧：nodeA_attn (Space-Node Attention) ==========
    im1 = ax1.imshow(attn_a, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_xlabel('Patch Index', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Space Node Index', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('Space-Node Attention', fontsize=13, fontweight='bold', pad=15)
    
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, aspect=30)
    cbar1.set_label('Attention Weight', fontsize=10, fontweight='bold', labelpad=8)
    cbar1.ax.tick_params(labelsize=9)
    
    # 设置边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.3)
    ax1.spines['bottom'].set_linewidth(1.3)
    ax1.tick_params(axis='both', which='major', labelsize=9, width=1.2)
    
    # ========== 右侧：nodeB_attn (Time-Node Attention) ==========
    im2 = ax2.imshow(attn_b, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax, origin='lower')
    ax2.set_xlabel('Space Node Index', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Patch Index', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('Patch-Node Attention', fontsize=13, fontweight='bold', pad=15)
    
  
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, aspect=30)
    cbar2.set_label('Attention Weight', fontsize=10, fontweight='bold', labelpad=8)
    cbar2.ax.tick_params(labelsize=9)
    
    # 设置边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.3)
    ax2.spines['bottom'].set_linewidth(1.3)
    ax2.tick_params(axis='both', which='major', labelsize=9, width=1.2)
    
    # 设置总标题
    if layer_idx is not None:
        fig.suptitle(f'Graph Attention Heatmaps', fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Graph Attention Heatmaps', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
    else:
        plt.show()


def visualize_embedding(embedding, title, save_path=None, csv_path=None, metric_column=None):
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()  # N, D

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=5,  # 降低困惑度以增强局部结构
        early_exaggeration=13,  # 增加早期夸大以促进聚类分离
        learning_rate='auto',  # 自动选择学习率
        init='pca',  # 使用 PCA 初始化更稳定
        metric='euclidean'  # 明确距离度量
    )
    emb_2d = tsne.fit_transform(embedding)
    plt.figure(figsize=(8, 6))

    # 如果提供了CSV文件路径和指标列名，则使用颜色映射
    if csv_path and metric_column:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查指定列是否存在
        if metric_column not in df.columns:
            raise ValueError(f"列 '{metric_column}' 在CSV文件中不存在")

        # 提取指标数据作为颜色值
        colors = df[metric_column].values

        # 使用颜色值绘制散点图，改为冷暖色系
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap='coolwarm')
        plt.colorbar(scatter)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1])

    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _convert_unix_to_relative_time(timestamps):
    """
    将Unix时间戳转换为相对时间（相对于第一个时间戳），单位固定为秒
    返回相对时间数组（秒）、时间单位标签和起始时间字符串
    """
    if len(timestamps) == 0:
        return timestamps, 'Time Step', None
    
    # 找到第一个有效时间戳作为基准
    base_time = timestamps[0]
    
    # 计算相对时间（秒）
    relative_times = timestamps - base_time
    # 固定使用秒作为单位
    time_label = 'Time (seconds)'
    
    # 格式化起始时间
    try:
        start_time_str = datetime.fromtimestamp(base_time).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError):
        # 如果时间戳超出范围，使用原始数值
        start_time_str = f'Base: {base_time:.0f}'
    
    return relative_times, time_label, start_time_str


def plot_patch_attention_diagnostics(record, title=None, save_path=None):
    """
    绘制单个 patch 内的流量曲线图。

    当前仅可视化流量随时间的变化，不再展示温度、熵或注意力权重。

    Args:
        record: 包含以下字段的字典：
            - x_value: [Tp] 原始流量数值
            - mask: [Tp] 掩码（>0.5 为有效观测）
            - indices: tuple (batch_idx, node_idx, patch_idx)，用于标题
            - timestamps: [Tp] 时间戳（可选，用于构造时间轴和估计观测间隔）
        title: 图表标题（可选）
        save_path: 保存路径（可选）
    """
    # 提取数据（仅保留流量和掩码等）
    x_value = record['x_value'].numpy() if isinstance(record['x_value'], torch.Tensor) else record['x_value']
    mask = record['mask'].numpy() > 0.5 if isinstance(record['mask'], torch.Tensor) else record['mask'] > 0.5
    indices = record['indices']
    
    # 获取时间戳（如果存在），用于时间轴和估计观测间隔
    if 'timestamps' in record:
        timestamps = record['timestamps'].numpy() if isinstance(record['timestamps'], torch.Tensor) else record['timestamps']
        use_timestamps = True
    else:
        timestamps = None
        use_timestamps = False

    # 只保留有效时间步
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return

    # 使用时间戳或索引作为x轴
    if use_timestamps:
        valid_timestamps = timestamps[valid_indices]
        valid_time_axis, _, _ = _convert_unix_to_relative_time(valid_timestamps)
        xlabel = 'Time Interval (s)'
    else:
        valid_time_axis = valid_indices
        xlabel = 'Time Interval'
    
    # 提取有效数据
    valid_x_value = x_value[valid_indices]

    # 设置科研级绘图风格
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.minor.width'] = 0.8
    plt.rcParams['ytick.minor.width'] = 0.8

    # ===== 仅绘制流量曲线 =====
    fig, ax = plt.subplots(figsize=(10, 4))

    # 绘制流量数据
    ax.plot(valid_time_axis, valid_x_value, marker='o', markersize=4.5,
            linewidth=2.2, color='#2E86AB',
            markerfacecolor='#2E86AB', markeredgecolor='white',
            markeredgewidth=1.2, alpha=0.95, zorder=3)
    ax.fill_between(valid_time_axis, 0, valid_x_value, alpha=0.15,
                    color='#2E86AB', zorder=1)

    # 设置x轴刻度为实际观测点的位置（真实间隔）
    # 如果数据点太多，适当采样显示刻度
    num_points = len(valid_time_axis)
    if num_points > 20:
        # 数据点多时，选择部分刻度显示
        step = max(1, num_points // 10)
        tick_positions = valid_time_axis[::step]
        tick_labels = [f'{t:.1f}' if abs(t) >= 1 else f'{t:.2f}' for t in tick_positions] if use_timestamps else [str(int(t)) for t in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
    else:
        # 数据点少时，显示所有刻度
        ax.set_xticks(valid_time_axis)
        if use_timestamps:
            tick_labels = [f'{t:.1f}' if abs(t) >= 1 else f'{t:.2f}' for t in valid_time_axis]
            ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
        else:
            ax.set_xticklabels([str(int(t)) for t in valid_time_axis], fontsize=9)

    # 简洁的轴标签
    ax.set_ylabel('Flow', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', labelpad=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.tick_params(axis='both', which='major', labelsize=9.5, width=1.2)
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.8)

    # 简洁的标题
    if title is None:
        title = 'Patch Flow'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    # ========== 保存或显示 ==========
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
    else:
        plt.show()

def save_all_visualizations(model, save_dir, epoch, csv_path=None, metric_column=None, max_patch_plots=16):
    save_dir = os.path.join(save_dir, f"epoch{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    # 保存 attention 图：将同一层的 nodeA_attn 和 nodeB_attn 放在同一个 PDF 文件中
    # max_layers = max(len(baselines.attn_logs['nodeA_attn']), len(baselines.attn_logs['nodeB_attn']))
    # for i in range(max_layers):
    #     attn_a = baselines.attn_logs['nodeA_attn'][i]
    #     attn_b = baselines.attn_logs['nodeB_attn'][i]
        
    #     plot_attention_heatmap(
    #         attn_a, 
    #         attn_b,
    #         save_path=os.path.join(save_dir, f"epoch_{epoch}_layer{i}_attention_heatmaps.pdf"),
    #         layer_idx=i
    #     )

    # # 保存 embedding
    # visualize_embedding(baselines.nodes_embedding, "Space-Node Embedding",
    #                     os.path.join(save_dir, f"epoch_{epoch}_nodeA_emb.png"),
    #                     csv_path=csv_path,
    #                     metric_column=metric_column)
    # visualize_embedding(baselines.patch_embedding, "NodeB (Patch) Embedding",
    #                     os.path.join(save_dir, f"epoch_{epoch}_nodeB_emb.png"))

    # 保存若干 patch 内的流量曲线图（如果模型中提供了 ttcn_logs）
    if getattr(model, 'ttcn_logs', None):
        patch_dir = os.path.join(save_dir, "patch_diagnostics")
        os.makedirs(patch_dir, exist_ok=True)

        # 从存储的 patch 数据中随机采样 max_patch_plots 个进行可视化
        import random
        available_logs = model.ttcn_logs
        if len(available_logs) > max_patch_plots:
            sampled_logs = random.sample(available_logs, max_patch_plots)
        else:
            sampled_logs = available_logs
        
        for idx, record in enumerate(sampled_logs):
            plot_patch_attention_diagnostics(
                record,
                save_path=os.path.join(patch_dir, f"epoch_{epoch}_patch_{idx}.pdf")
            )
