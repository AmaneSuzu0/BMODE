import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


# ========================== GraphAttentionLayer (from GAT_ODE_irregular) ==========================
class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, layer_idx):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # Pre-Norm
        self.norm_edges_in = nn.LayerNorm(dim)
        self.norm_nodeA_in = nn.LayerNorm(dim)
        self.norm_nodeB_in = nn.LayerNorm(dim)

        # 边更新
        self.edge_out = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # 节点A更新
        self.nodeA_q = nn.Linear(dim, dim)
        self.nodeA_k = nn.Linear(dim, dim)
        self.nodeA_v = nn.Linear(dim, dim)
        self.nodeA_out = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # 节点B更新
        self.nodeB_q = nn.Linear(dim, dim)
        self.nodeB_k = nn.Linear(dim, dim)
        self.nodeB_v = nn.Linear(dim, dim)
        self.nodeB_out = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, edges, nodeA, nodeB, mask=None):
        """
        edges: [B, N, T, dim]
        nodeA: [B, N, dim]
        nodeB: [B, T, dim]
        mask:  [B, N, T] bool, True for valid positions
        """
        B, N, T, _ = edges.shape

        # Normalize
        edges = self.norm_edges_in(edges)
        nodeA = self.norm_nodeA_in(nodeA)
        nodeB = self.norm_nodeB_in(nodeB)

        # ---- 1. Edge update ----
        edges_flat = edges.reshape(B, N * T, -1)
        nodeA_expand = nodeA.unsqueeze(2).expand(-1, -1, T, -1).reshape(B, N * T, -1)
        nodeB_expand = nodeB.unsqueeze(1).expand(-1, N, -1, -1).reshape(B, N * T, -1)
        concat = torch.cat([edges_flat, nodeA_expand, nodeB_expand], dim=-1)
        new_edges = self.edge_out(concat).reshape(B, N, T, self.dim)

        # ---- 2. NodeA update (aggregate over time) ----
        qA = self.nodeA_q(nodeA).unsqueeze(2)                     # [B,N,1,dim]
        kA = self.nodeA_k(edges)                                  # [B,N,T,dim]
        vA = self.nodeA_v(edges)
        score = torch.einsum("bnqd,bnkd->bnqk", qA, kA).squeeze(2) / (self.temperature+1e-4)  # [B,N,T]
        if mask is not None:
            score = score.masked_fill(~mask.bool(), -1e9)
        probA = F.softmax(score, dim=-1)
        outA = torch.einsum("bnt,bntd->bnd", probA, vA)           # [B,N,dim]
        new_nodeA = self.nodeA_out(torch.cat([nodeA, outA], dim=-1))

        # ---- 3. NodeB update (aggregate over space) ----
        edges_TN = edges.transpose(1, 2)                          # [B,T,N,dim]
        qB = self.nodeB_q(nodeB).unsqueeze(2)                     # [B,T,1,dim]
        kB = self.nodeB_k(edges_TN)                               # [B,T,N,dim]
        vB = self.nodeB_v(edges_TN)
        scoreB = torch.einsum("btqd,btkd->btqk", qB, kB).squeeze(2) / (self.temperature+1e-4)  # [B,T,N]
        if mask is not None:
            mask_TN = mask.transpose(1, 2)                        # [B,T,N]
            scoreB = scoreB.masked_fill(~mask_TN.bool(), -1e9)
        probB = F.softmax(scoreB, dim=-1)
        outB = torch.einsum("btn,btnd->btd", probB, vB)           # [B,T,dim]
        new_nodeB = self.nodeB_out(torch.cat([nodeB, outB], dim=-1))

        return new_edges, new_nodeA, new_nodeB, probA, probB


# ========================== New Encoder using GraphAttentionLayer ==========================
class BipartiteGraphEncoder(nn.Module):
    """
    Encoder based on bipartite graph attention.
    Input:  [B, L, N, C]   (batch, history_len, nodes, features)
    Output: [B, N, D]      (batch, nodes, output_dim)
    """
    def __init__(self, input_dim, dim, num_nodes, seq_len, patch_size, n_layers=3, output_dim=None):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.output_dim = output_dim if output_dim is not None else dim
        self.patch_size = patch_size
        self.patch_num = seq_len // patch_size
        # Map input features to edge features
        self.input_proj = nn.Linear(input_dim * self.patch_size, dim)

        # Learnable NodeA (spatial) and NodeB (temporal) embeddings
        self.nodeA = nn.Parameter(torch.randn(num_nodes, dim))
        self.nodeB = nn.Parameter(torch.randn(self.patch_num, dim))

        # Stack of graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(dim, i) for i in range(n_layers)
        ])

        self.edges_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(self.dim * self.patch_num, self.output_dim)

        # Initialize
        nn.init.xavier_uniform_(self.nodeA)
        nn.init.xavier_uniform_(self.nodeB)

    def forward(self, history_data):
        """
        history_data: [B, L, N, C]
        Returns: [B, N, output_dim]
        """
        B, L, N, C = history_data.shape
        assert N == self.num_nodes and L == self.seq_len
        assert L % self.patch_num == 0
        # Permute to [B, N, L, C] and project to edge features

        edges = history_data.permute(0, 2, 1, 3).contiguous()  # [B, N, L, C]
        edges = edges.reshape(B, N, self.patch_num, C * self.patch_size)
        edges = self.input_proj(edges)  # [B, N, num_patches, dim]

        # Initial node features
        nodeA = self.nodeA.unsqueeze(0).expand(B, -1, -1)  # [B,N,dim]
        nodeB = self.nodeB.unsqueeze(0).expand(B, -1, -1)  # [B,L,dim]

        # All positions are valid (no mask needed)
        mask = torch.ones(B, N, self.patch_num, dtype=torch.bool, device=history_data.device)

        # Multi-layer graph attention
        for layer in self.graph_layers:
            new_edges, new_nodeA, new_nodeB, _, _ = layer(edges, nodeA, nodeB, mask)
            edges = edges + new_edges
            nodeA = nodeA + new_nodeA
            nodeB = nodeB + new_nodeB

        edges = self.edges_norm(edges)   # [B,N,L,dim]


        z0 = edges.reshape(B, N, -1)         # [B,N,L*dim]
        # Project to target dimension
        z0 = self.out_proj(z0)           # [B,N,output_dim]
        return z0


# ========================== ODE Decoder (from DCRNN) ==========================
class ODEFunc(nn.Module):
    """Multi-mode dynamics with weight evolution via logits.

    Supports two modes:
      - soft (default):  dz/dt = Σ w_i * f_i(z),  w = softmax(logits)
      - hard:            dz/dt = f_k(z) where k = argmax(logits),
                         using Straight-Through Estimator (STE) for training.
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
        # 网络输出 d(logits)/dt
        self.logits_net = nn.Sequential(
            nn.Linear(dim + num_modes, dim),
            nn.Tanh(),
            nn.Linear(dim, num_modes)
        )


    def forward(self, t, state):
        z, logits = state
        w = F.softmax(logits, dim=-1)
        # dz/dt
        dz_all = torch.stack([f(z) for f in self.funcs], dim=1)
        dz = (w.unsqueeze(-1) * dz_all).sum(dim=1)
        # d(logits)/dt
        dlogits = self.logits_net(torch.cat([z, logits], dim=-1))
        return dz, dlogits


class SmoothODEDecoder(nn.Module):
    """Multi-mode ODE decoder with smooth weight switching"""
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
        outputs = [f(z_sample) for f in self.ode_func.funcs]  # list of [B, D]
        outputs = torch.stack(outputs, dim=1)  # [B, M, D]
        outputs_norm = F.normalize(outputs, dim=-1)
        # 计算不同模式间的余弦相似度矩阵，去掉对角线（自己）
        sim_mat = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))  # [B, M, M]
        mask = 1 - torch.eye(self.num_modes, device=z_sample.device)  # 非对角线为1
        off_diag = sim_mat * mask
        # 惩罚正相关（相似度过高），但对负相关不惩罚（可以允许不同方向）
        diversity_loss = torch.relu(off_diag).mean()
        return diversity_loss

    def forward(self, z0, t_pred, return_z0=False):
        """
        z0: (B, N, D)
        t_pred: (T_pred,) time points (must start with 0)
        return_z0: 是否返回初始状态用于计算正则化
        Returns: flow (T_pred, B, N, 1) 或 (flow, z0_flat) 如果 return_z0=True
        """
        B, N, D = z0.shape
        z_flat = z0.reshape(B * N, D)
        w0 = self.mode_net(z_flat)                     # (B*N, num_modes)
        state = (z_flat, w0)
        sol = odeint(self.ode_func, state, t_pred, method='euler')
        z_traj = sol[0]   # (len(t_pred), B*N, D)
        # Reshape: (T_pred, B, N, D)
        z_traj = z_traj.permute(1, 0, 2).reshape(B, N, len(t_pred), D).permute(2, 0, 1, 3)
        flow = self.readout(z_traj)                    # (T_pred, B, N, 1)
        
        if return_z0:
            return flow, z_flat
        return flow


# ========================== Modified DCRNN with BipartiteGraphEncoder ==========================
class GATODE_r(nn.Module):
    """DCRNN with bipartite graph attention encoder and neural ODE decoder"""
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        # Required parameters (must be present in model_kwargs)
        self.dim = int(model_kwargs.get("dim"))
        self.num_nodes = int(model_kwargs.get("num_nodes"))
        self.seq_len = int(model_kwargs.get("seq_len", 12))
        self.horizon = int(model_kwargs.get("horizon", 12))
        self.input_dim = int(model_kwargs.get("input_dim", 3))

        # Encoder
        self.encoder = BipartiteGraphEncoder(
            input_dim=self.input_dim,
            dim=model_kwargs.get("dim"),
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            n_layers=model_kwargs.get("gat_layers", 2),
            patch_size=model_kwargs.get("patch_size", 1),
            output_dim=self.dim   # must match decoder's expected dim
        )

        # Decoder
        self.decoder = SmoothODEDecoder(
            dim=model_kwargs.get("dim"),
            num_modes=model_kwargs.get("num_modes", 3),
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, **kwargs):
        """
        history_data: [B, L, N, C]
        Returns: [B, H, N, 1] predictions
        """
        train = kwargs.get('train', False)
        
        z0 = self.encoder(history_data)                # [B, N, D]
        t_pred = torch.linspace(0, 1, self.horizon, device=z0.device)
        
        if train and self.decoder.num_modes > 1:
            outputs, z_flat = self.decoder(z0, t_pred, return_z0=True)  # [H, B, N, 1], [B*N, D]
            outputs = outputs.transpose(0, 1)          # [B, H, N, 1]
            # 计算正交性损失
            ortho_loss = self.decoder.compute_diversity_loss(z_flat)
            return {'prediction': outputs, 'ortho_loss': ortho_loss}
        else:
            outputs = self.decoder(z0, t_pred)         # [H, B, N, 1]
            outputs = outputs.transpose(0, 1)          # [B, H, N, 1]
            return outputs