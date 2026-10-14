import torch

class ZhuzhouConfig:
    # patch参数
    PATCH_WINDOW = 200  # 观测窗口时间
    PATCH_STRIDE = 200  # 观测窗口时间步长
    OBS_PATCH_LEN = ((3600 - PATCH_WINDOW) // PATCH_STRIDE) + 1   # patch大小
    # 数据参数
    MEAN_FLOW = 3.399886161402731  # 均值
    STD_FLOW = 4.157979465894482  # 标准差
    MEAN_PERIOD = 110.05734090100657  # 周期均值
    STD_PERIOD = 19.628382740376114  # 周期标准差
    # 设备设置
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 模型参数
    MODEL_PARAMS = {
        'dim': 32,
        'node_dim': 32,
        'num_nodes': 620,
        'n_layers': 3,
        'num_modes': 3,
        'patch_len': OBS_PATCH_LEN,
        'device': DEVICE,
    }

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 200
    LEARNING_RATE = 0.001
    PATIENCE = 70  # 早停耐心值
    MIN_DELTA = 0.0001  # 早停最小改进
    WEIGHT_DECAY = 0.0001  # 权重衰减
    ORTHO_WEIGHT = None  # 多样性损失权重 (hard_switch=True时建议≥0.1, soft模式可用None)
    # 数据集划分
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    # 日志和保存
    LOG_DIR = "./checkpoints/zhuzhou"
    SAVE_DIR = "./checkpoints/zhuzhou/saved_models/"
    DATA_DIR = f"./datasets/zhuzhou/data_patch_{PATCH_WINDOW}_{PATCH_STRIDE}.pkl"
    VISUAL_DIR = "./visual_results/zhuzhou"
    # 学习率策略
    LR_DECAY_STEP = 50  # 每10个epoch衰减一次
    LR_DECAY_RATE = 0.5 # 学习率衰减为原来的0.5倍

    # 节点中心性
    NODE_CENTRALITY_PATH = './datasets/zhuzhou/node_centralities.csv'
    NODE_CENTRALITY_METRIC = 'combined'
    # 空间关系矩阵
    DISTANCE_GEO_PATH = './datasets/zhuzhou/distance_geo.npy'
    REACHABILITY_PATH = './datasets/zhuzhou/reachability.npy'
    # 随机种子
    SEED = 42
