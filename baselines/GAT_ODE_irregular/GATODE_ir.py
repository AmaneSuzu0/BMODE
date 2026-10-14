import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import random

class DTTCN(nn.Module):
    def __init__(
        self,
        dim,
        eps: float = 1e-6,
        sharp_temperature: float = 0.5
    ):
        """
            DTTCN: 返回两个聚合版本
            - 平均聚合：直接对patch内时间步进行平均
            - z-score注意力聚合：基于z-score的softmax注意力，关注偏离平均趋势的时间步

        Args:
            dim: 输入每时间步 embedding 的维度（x 的最后一维）
            eps: 数值稳定项
            sharp_temperature: softmax温度，<1时让分布更尖锐（默认0.5）
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.sharp_temperature = sharp_temperature

    def forward(self, x, x_mask, x_value, timestamps=None):
        """
        Args:
            x: [B, N, P, Tp, dim]  -- embedding for each time step
            x_mask: [B, N, P, Tp]  -- 0/1 mask (float/int)
            x_value: [B, N, P, Tp] -- 原始流量数值（用于计算 deviation）
            timestamps: optional (仅用于日志)
        Returns:
            agg_mean: [B, N, P, dim] - 直接平均聚合
            agg_sharp: [B, N, P, dim] - z-score注意力聚合（关注偏离平均趋势的）
        """
        B, N, P, Tp, dim = x.shape
        device = x.device
        assert dim == self.dim, f"dim mismatch: expected {self.dim}, got {dim}"

        mask = x_mask.float()
        valid_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B,N,P,1]

        # ---------- 1) 基本 patch 统计量 ----------
        mean = (x_value * mask).sum(dim=-1, keepdim=True) / valid_count  # [B,N,P,1]
        var = (((x_value - mean) ** 2) * mask).sum(dim=-1, keepdim=True) / valid_count
        std = torch.sqrt(var + self.eps)  # [B,N,P,1]

        # ---------- 2) 直接平均聚合 ----------
        agg_mean = (x * mask.unsqueeze(-1)).sum(dim=-2) / valid_count  # [B,N,P,dim]

        # ---------- 3) deviation score (z-score 的绝对值) ----------
        z = torch.abs((x_value - mean) / (std + self.eps))  # [B,N,P,Tp]

        # ---------- 4) z-score注意力聚合（关注偏离平均趋势的） ----------
        large_neg = -1e9
        scores = z.masked_fill(mask == 0, large_neg)  # [B,N,P,Tp]
        attn_sharp_logits = scores / (self.sharp_temperature + self.eps)  # [B,N,P,Tp]
        attn_sharp = F.softmax(attn_sharp_logits, dim=-1) * mask  # [B,N,P,Tp]
        attn_sharp = attn_sharp / (attn_sharp.sum(dim=-1, keepdim=True).clamp(min=self.eps))
        attn_sharp_e = attn_sharp.unsqueeze(-1)  # [B,N,P,Tp,1]
        agg_sharp = torch.sum(attn_sharp_e * x, dim=-2)  # [B,N,P,dim]

        return agg_mean, agg_sharp


class ODEFunc(nn.Module):
    """Multi-mode dynamics with weight evolution via logits.
    """
    def __init__(self, dim, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Linear(dim, dim)
            ) for _ in range(num_modes)
        ])
        self.logits_net = nn.Sequential(
            nn.Linear(dim + num_modes, dim),
            nn.Tanh(),
            nn.Linear(dim, num_modes)
        )

    def forward(self, t, state):
        z, logits = state
        w = F.softmax(logits, dim=-1)
        dz_all = torch.stack([f(z) for f in self.funcs], dim=1)
        dz = (w.unsqueeze(-1) * dz_all).sum(dim=1)

        dlogits = self.logits_net(torch.cat([z, logits], dim=-1))
        return dz, dlogits


class TemporalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(TemporalEmbedding, self).__init__()
        self.time_day = nn.Linear(1, emb_dim)
        self.time_week = nn.Embedding(7, emb_dim)

    def forward(self, x, mask=None):
        day_emb = x[..., 1]
        week_emb = x[..., 0].long()
        time_day = self.time_day(day_emb.unsqueeze(-1))
        time_week = self.time_week(week_emb)
        tem_emb = time_day + time_week
        if mask is not None:
            tem_emb = tem_emb * mask.unsqueeze(-1)
        return tem_emb

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, node_dim, layer_idx):
        super().__init__()
        self.dim = dim
        self.node_dim = node_dim
        self.layer_idx = layer_idx
        # 新增：Pre-Norm 层（注意力计算前的归一化）
        self.norm_edges_in = nn.LayerNorm(dim)
        self.norm_nodeA_in = nn.LayerNorm(node_dim)
        self.norm_nodeB_in = nn.LayerNorm(node_dim)

        # ---------- 边更新 ----------
        self.edge_out = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # ---------- 节点A 更新 ----------
        self.nodeA_q = nn.Linear(node_dim, node_dim)
        self.nodeA_k = nn.Linear(dim, node_dim)
        self.nodeA_v = nn.Linear(dim, node_dim)
        self.nodeA_out = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

        # ---------- 节点B 更新 ----------
        self.nodeB_q = nn.Linear(node_dim, node_dim)
        self.nodeB_k = nn.Linear(dim, node_dim)
        self.nodeB_v = nn.Linear(dim, node_dim)
        self.nodeB_out = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

        self.temperature = nn.Parameter(torch.ones(1))  # 初始缩放参数

    def forward(self, edges, nodeA, nodeB, patch_mask=None):
        """
        Args:
            edges: [B, N, T, dim]
            nodeA: [B, N, node_dim]
            nodeB: [B, T, node_dim]
            patch_mask: [B, N, T] 或 None
        """
        B, N, T, _ = edges.shape

        edges = self.norm_edges_in(edges)  # 边特征先归一化
        nodeA = self.norm_nodeA_in(nodeA)  # NodeA先归一化
        nodeB = self.norm_nodeB_in(nodeB)  # NodeB先归一化

        # >>>> 1. 边更新
        edges_flat = edges.reshape(B, N * T, -1)  # [B, N*T, dim]
        nodeA_i = nodeA.unsqueeze(2).expand(-1, -1, T, -1).reshape(B, N * T, -1)  # [B, N*T, node_dim]
        nodeB_j = nodeB.unsqueeze(1).expand(-1, N, -1, -1).reshape(B, N * T, -1)  # [B, N*T, node_dim]
        nodeAB_concat = torch.cat([edges_flat, nodeA_i, nodeB_j], dim=-1)  # [B, N*T, 3*node_dim]
        new_edges = self.edge_out(nodeAB_concat).reshape(B, N, T, self.dim)  # [B, N, T, dim]

        # >>>> 2. A 类节点更新（在时间维上使用 patch_mask）
        qA = self.nodeA_q(nodeA).unsqueeze(2)  # [B, N, 1, node_dim]
        # 使用原本的edges
        kA = self.nodeA_k(edges)              # [B, N, T, node_dim]
        vA = self.nodeA_v(edges)              # [B, N, T, node_dim]
        # # 使用更新后的edges
        # kA = self.nodeA_k(new_edges)              # [B, N, T, node_dim]
        # vA = self.nodeA_v(new_edges)              # [B, N, T, node_dim]

        score = torch.einsum("bnqd,bnkd->bnqk", qA, kA).squeeze(2) / self.temperature  # [B, N, T]
        if patch_mask is not None:
            score = score.masked_fill(~patch_mask.bool(), -1e9)
        prob_A = F.softmax(score, dim=-1)
        outA = torch.einsum("bnt,bntd->bnd", prob_A, vA)  # [B, N, node_dim]
        new_nodeA = self.nodeA_out(torch.cat([nodeA, outA], dim=-1))

        # >>>> 3. B 类节点更新（对每个时间节点，沿空间维度掩码）
        edges_TN = edges.transpose(1, 2)  # [B, T, N, dim] 使用原edges
        # edges_TN = new_edges.transpose(1, 2)  # [B, T, N, dim] 使用更新后的edges
        qB = self.nodeB_q(nodeB).unsqueeze(2)  # [B, T, 1, node_dim]
        kB = self.nodeB_k(edges_TN)            # [B, T, N, node_dim]
        vB = self.nodeB_v(edges_TN)            # [B, T, N, node_dim]

        score_B = torch.einsum("btqd,btkd->btqk", qB, kB).squeeze(2) / self.temperature  # [B, T, N]
        if patch_mask is not None:
            patch_mask_TN = patch_mask.transpose(1, 2)  # [B, T, N]
            score_B = score_B.masked_fill(~patch_mask_TN.bool(), -1e9)
        prob_B = F.softmax(score_B, dim=-1)
        outB = torch.einsum("btn,btnd->btd", prob_B, vB)  # [B, T, node_dim]
        new_nodeB = self.nodeB_out(torch.cat([nodeB, outB], dim=-1))

        return new_edges, new_nodeA, new_nodeB, prob_A, prob_B


class GatedFusionLayer(nn.Module):
    """门控融合层：融合两路特征（mean/sharp），支持两种门控模式"""
    def __init__(self, dim, fusion_mode="gate_additive"):
        super().__init__()
        self.dim = dim
        self.fusion_mode = fusion_mode
        
        if fusion_mode == "gate_additive":
            # 加法门控（GLU变体）
            self.gate = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Sigmoid()
            )
            self.residual = nn.Linear(dim, dim)
        elif fusion_mode == "gate_multiplicative":
            # 乘法门控
            self.gate_mean = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
            self.gate_sharp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        elif fusion_mode == "gate_attention":
            # 注意力门控
            self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")
        
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(self, feat_mean, feat_sharp):
        """
        Args:
            feat_mean: [B, N, P, dim] - 平均聚合支路特征
            feat_sharp: [B, N, P, dim] - 尖锐聚合支路特征
        Returns:
            fused_feat: [B, N, P, dim] - 门控融合后的特征
        """
        B, N, P, D = feat_mean.shape
        
        if self.fusion_mode == "gate_additive":
            # 加法门控：gate = σ([mean; sharp])，output = gate * mean + (1 - gate) * sharp
            concat_feat = torch.cat([feat_mean, feat_sharp], dim=-1)
            gate = self.gate(concat_feat)  # [B, N, P, D]
            fused = gate * feat_mean + (1 - gate) * feat_sharp
        
        elif self.fusion_mode == "gate_multiplicative":
            # 乘法门控：output = (gate_mean * mean) * (gate_sharp * sharp) + mean + sharp
            gate_m = self.gate_mean(feat_mean)
            gate_s = self.gate_sharp(feat_sharp)
            fused = (gate_m * feat_mean) * (gate_s * feat_sharp) + feat_mean + feat_sharp
        
        elif self.fusion_mode == "gate_attention":
            # 注意力门控：用mean查询sharp，再门控融合
            mean_flat = feat_mean.reshape(B * N, P, D)
            sharp_flat = feat_sharp.reshape(B * N, P, D)
            
            # 自注意力增强
            attn_out, _ = self.attn(mean_flat, sharp_flat, sharp_flat)
            attn_out = attn_out.reshape(B, N, P, D)
            
            # 门控融合
            gate = self.gate(attn_out)
            fused = gate * attn_out + (1 - gate) * feat_mean
        
        # 残差 + 层归一化 + FFN
        fused = self.norm(fused + feat_mean + feat_sharp)
        fused = self.norm(fused + self.ffn(fused))
        
        return fused


class ODEDecoder(nn.Module):
    """ODE decoder operating on flat [B*N, dim] input"""
    def __init__(self, dim, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.ode_func = ODEFunc(dim, num_modes)
        self.mode_net = nn.Sequential(
            nn.Linear(dim, num_modes)
        )
        self.readout = nn.Sequential(
            nn.Linear(dim, 1),
        )

    def compute_diversity_loss(self, z_sample):
        """鼓励模式输出不同，但不强制正交"""
        if self.num_modes <= 1:
            return 0.0
        outputs = [f(z_sample) for f in self.ode_func.funcs]  # list of [B*N, D]
        outputs = torch.stack(outputs, dim=1)  # [B*N, M, D]
        outputs_norm = F.normalize(outputs, dim=-1)
        # 计算不同模式间的余弦相似度矩阵，去掉对角线（自己）
        sim_mat = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))  # [B*N, M, M]
        mask = 1 - torch.eye(self.num_modes, device=z_sample.device)  # 非对角线为1
        off_diag = sim_mat * mask
        # 惩罚正相关（相似度过高），但对负相关不惩罚（可以允许不同方向）
        diversity_loss = torch.relu(off_diag).mean()
        return diversity_loss

    def forward(self, z0_flat, t_eval, return_z0=False):
        """
        Args:
            z0_flat: [B*N, dim]
            t_eval:  [T,] sorted time grid (must start at 0)
            return_z0: 是否返回初始状态用于计算正则化
        Returns:
            preds: [B*N, T]  或  (preds, z0_flat) 如果 return_z0=True
        """
        w0 = self.mode_net(z0_flat)
        state = (z0_flat, w0)
        sol = odeint(self.ode_func, state, t_eval,
                     method='euler')
        z_traj = sol[0]                     # [T, B*N, dim]
        flow = self.readout(z_traj)         # [T, B*N, 1]
        flow = flow.squeeze(-1)             # [T, B*N]
        preds = flow.permute(1, 0)          # [B*N, T]

        if return_z0:
            return preds, z0_flat
        return preds


class GATODE_ir(nn.Module):
    def __init__(self, dim, num_nodes, n_layers, patch_len, node_dim, device, num_modes,
                 sharp_temperature=1.0, fusion_mode="gate_additive",):
        """
        Args:
            dim: embedding维度
            num_nodes: 节点数量
            n_layers: 图注意力层数
            patch_len: patch长度
            node_dim: 节点embedding维度
            device: 设备
            sharp_temperature: softmax温度
            fusion_mode: 门控融合模式 ("gate_additive"/"gate_multiplicative"/"gate_attention")
        """
        super().__init__()
        self.patch_len = patch_len
        self.time_embedding = TemporalEmbedding(dim)
        self.flow_embedding = nn.Linear(1, dim)
        self.period_embedding = nn.Linear(1, dim)
        self.n_layers = n_layers
        self.dim = dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.device = device

        self.ttcn = DTTCN(dim=dim, sharp_temperature=sharp_temperature)
        self.edges_norm_sharp = nn.LayerNorm(dim)
        self.edges_norm_mean = nn.LayerNorm(dim)
        
        # ========== 改动1: 可学习参数化的NodeA/NodeB嵌入 ==========
        # NodeA: 可学习的节点嵌入（两路独立）
        self.nodeA_emb_mean = nn.Parameter(torch.randn(num_nodes, node_dim))  # [N, node_dim]
        self.nodeA_emb_sharp = nn.Parameter(torch.randn(num_nodes, node_dim))  # [N, node_dim]
        # NodeA投影层（增强表达）
        self.nodeA_proj_mean = nn.Linear(node_dim, node_dim)
        self.nodeA_proj_sharp = nn.Linear(node_dim, node_dim)
        
        # NodeB: 可学习的Patch时间嵌入（两路独立）
        self.nodeB_emb_mean = nn.Parameter(torch.randn(patch_len, node_dim))  # [P, node_dim]
        self.nodeB_emb_sharp = nn.Parameter(torch.randn(patch_len, node_dim))  # [P, node_dim]
        # NodeB投影层（增强表达）
        self.nodeB_proj_mean = nn.Linear(node_dim, node_dim)
        self.nodeB_proj_sharp = nn.Linear(node_dim, node_dim)

        # 两路独立的图注意力网络
        self.graph_layers_mean = nn.ModuleList(
            [GraphAttentionLayer(dim, node_dim, i) for i in range(n_layers)]
        )
        self.graph_layers_sharp = nn.ModuleList(
            [GraphAttentionLayer(dim, node_dim, i) for i in range(n_layers)]
        )
           
        # ========== 改动2: 门控融合层（替代原有的简单拼接） ==========
        self.fusion_layer = GatedFusionLayer(dim, fusion_mode=fusion_mode)

        # ODE解码器
        self.ode_decoder = ODEDecoder(dim=dim, num_modes=num_modes,)
        self.time_scale = 3600.0  # 时间戳归一化（秒→小时）

        # 初始化可学习嵌入
        nn.init.xavier_uniform_(self.nodeA_emb_mean)
        nn.init.xavier_uniform_(self.nodeA_emb_sharp)
        nn.init.xavier_uniform_(self.nodeB_emb_mean)
        nn.init.xavier_uniform_(self.nodeB_emb_sharp)

    def forward(self, x, y, x_mask, y_mask, **kwargs):
        """
        :param x:  [B, N, P, Tp, 5]
        :param y: [B, N, P, Tp, 5]
        :param x_mask: [B, N, P, Tp]
        :param y_mask: [B, N, P, Tp]
        :param kwargs: 支持 train=True 时返回 ortho_loss
        :return: 预测流量 [B, N, P, Tp] 或 dict {'prediction': ..., 'ortho_loss': ...}
        """
        train = kwargs.get('train', False)
        # 在转换前先提取时间戳，使用 clone() 创建独立副本，保持 float64 精度
        x_timestamps = x[..., 0].clone()  # [B, N, P_obs, Tp] - 真实时间戳（保持 float64）
        y_timestamps = y[..., 0].clone()  # [B, N, P_pred, Tp] - 保持 float64，避免大数值减法精度丢失

        # 将 x 和 y 转换为 float32（参与计算的部分）
        x = x.float()
        y = y.float()
        
        b, n, p, pl, _ = x.shape
        x_value = x[..., 4]  # [B, N, P, Tp]
        # patch 级掩码：每个 patch 是否至少有一个有效时间步
        patch_valid = (x_mask.sum(dim=-1) > 0)  # [B, N, P] bool
        
        # 1. 时间嵌入和流量嵌入
        x_tem_emb = self.time_embedding(x[..., 1:3], x_mask)  # [B, N, P, Tp, dim]
        x_period_emb = self.period_embedding(x[..., 3].unsqueeze(-1)) * x_mask.unsqueeze(-1)  # [B, N, P, Tp, dim]
        x_value_emb = self.flow_embedding(x_value.unsqueeze(-1)) * x_mask.unsqueeze(-1)  # [B, N, P, Tp, dim]
        x_emb = x_tem_emb + x_period_emb + x_value_emb  # [B, N, P, Tp, dim]
        
        # 2. 应用DTTCN得到两个聚合版本
        edges_mean, edges_sharp = self.ttcn(x_emb, x_mask, x_value, timestamps=x_timestamps)  # 都是 [B, N, P, dim]

        # 3. 生成节点特征（可学习参数化版本）
        # 3.1 NodeA: 可学习嵌入 + 投影（两路独立）
        nodeA_mean = self.nodeA_proj_mean(self.nodeA_emb_mean)  # [N, node_dim]
        nodeA_mean = nodeA_mean.unsqueeze(0).expand(b, -1, -1)  # [B, N, node_dim]
        
        nodeA_sharp = self.nodeA_proj_sharp(self.nodeA_emb_sharp)  # [N, node_dim]
        nodeA_sharp = nodeA_sharp.unsqueeze(0).expand(b, -1, -1)  # [B, N, node_dim]
        
        # 3.2 NodeB: 可学习嵌入 + 投影（两路独立）
        nodeB_mean = self.nodeB_proj_mean(self.nodeB_emb_mean)  # [P, node_dim]
        nodeB_mean = nodeB_mean.unsqueeze(0).expand(b, -1, -1)  # [B, P, node_dim]
        
        nodeB_sharp = self.nodeB_proj_sharp(self.nodeB_emb_sharp)  # [P, node_dim]
        nodeB_sharp = nodeB_sharp.unsqueeze(0).expand(b, -1, -1)  # [B, P, node_dim]


        # 4. 两路分别进行多层图注意力更新
        # 第一路：处理平均聚合结果
        for i in range(self.n_layers):
            mask_for_layer = patch_valid if i == 0 else None
            new_edges, new_nodeA, new_nodeB, nodeA_attn, nodeB_attn = self.graph_layers_mean[i](
                edges_mean, nodeA_mean, nodeB_mean, mask_for_layer
            )
            edges_mean = edges_mean + new_edges
            nodeA_mean = nodeA_mean + new_nodeA
            nodeB_mean = nodeB_mean + new_nodeB
        edges_mean = self.edges_norm_mean(edges_mean)
        # 第二路：处理z-score注意力聚合结果
        for i in range(self.n_layers):
            mask_for_layer = patch_valid if i == 0 else None
            new_edges, new_nodeA, new_nodeB, nodeA_attn, nodeB_attn = self.graph_layers_sharp[i](
                edges_sharp, nodeA_sharp, nodeB_sharp, mask_for_layer
            )
            edges_sharp = edges_sharp + new_edges
            nodeA_sharp = nodeA_sharp + new_nodeA
            nodeB_sharp = nodeB_sharp + new_nodeB
        edges_sharp = self.edges_norm_sharp(edges_sharp)
        # 5. 门控融合两路edges特征（替代原有的拼接+线性层）
        edges = self.fusion_layer(edges_mean, edges_sharp)  # [B, N, P, dim]

        # ===== 6. ODE解码器 =====
        p_pred = y.shape[2]
        y_mask_bool = y_mask.bool()

        # 6.1 聚合patch特征到节点级z0（masked mean pooling）
        patch_valid_obs = patch_valid.float()  # [B, N, P_obs]
        valid_count = patch_valid_obs.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        z0 = (edges * patch_valid_obs.unsqueeze(-1)).sum(dim=2) / valid_count  # [B, N, dim]
        z0_flat = z0.reshape(b * n, self.dim)  # [B*N, dim]

        # 6.2 参考时间戳：每个节点最早有效预测时间戳（float64）
        y_ts_masked = y_timestamps.clone()
        y_ts_masked[~y_mask_bool] = float('inf')
        ref_t, _ = y_ts_masked.view(b, n, -1).min(dim=-1)  # [B, N]
        ref_t = torch.where(torch.isinf(ref_t), torch.zeros_like(ref_t), ref_t)

        # 6.3 相对时间戳：首个有效预测点从0开始（float64 → float32）
        rel_ts = (y_timestamps - ref_t.unsqueeze(-1).unsqueeze(-1)) / self.time_scale
        rel_ts = rel_ts.float()  # [B, N, P_pred, Tp], float32

        # 6.4 构建时间网格：只取有效预测点，去重排序
        rel_ts_valid = rel_ts[y_mask_bool]
        if rel_ts_valid.numel() == 0:
            return torch.zeros(b, n, p_pred, pl, device=z0.device)

        # 四舍五入到1e-4小时精度（≈0.36秒），合并浮点精度导致的近重复值
        rel_ts_rounded = torch.round(rel_ts_valid * 1e4) / 1e4
        t_grid = torch.unique(torch.sort(rel_ts_rounded)[0])
        if t_grid[0].item() > 1e-8:
            t_grid = torch.cat([torch.zeros(1, device=z0.device, dtype=t_grid.dtype), t_grid])

        # 6.5 运行ODE（训练时返回z0用于计算正交损失）
        if train and self.ode_decoder.num_modes > 1:
            pred_flat, z0_for_ortho = self.ode_decoder(z0_flat, t_grid, return_z0=True)  # [B*N, T_grid]
            ortho_loss = self.ode_decoder.compute_diversity_loss(z0_for_ortho)
        else:
            pred_flat = self.ode_decoder(z0_flat, t_grid)  # [B*N, T_grid]

        # 6.6 查表映射回patch格式
        T_grid = len(t_grid)
        pos = torch.searchsorted(t_grid, rel_ts)  # [B, N, P_pred, Tp]
        pos = pos.clamp(0, T_grid - 1)

        pred_flat_1d = pred_flat.reshape(-1)  # [B*N*T_grid]
        flat_idx = (torch.arange(b * n, device=z0.device).view(b, n, 1, 1) * T_grid + pos)
        output = pred_flat_1d[flat_idx]  # [B, N, P_pred, Tp]

        # 6.7 掩码无效预测位置
        output = output * y_mask.float()

        if train and self.ode_decoder.num_modes > 1:
            return {'prediction': output, 'ortho_loss': ortho_loss}
        return output
