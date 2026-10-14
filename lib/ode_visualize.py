"""
ODE trajectory visualization module for multi-vector-field analysis.

Provides:
  1. record_ode_trajectory_euler — manual Euler integration recording weights + f_outputs
  2. run_single_field_integration — force single-field ODE for ablation study
  3. plot_field_weights_vs_flow — dual subplot: weights + true flow
  4. plot_weight_flow_overlay — flow curve with dominant field background
  5. plot_vector_field_tsne — t-SNE of vector field outputs colored by field
  6. plot_field_ablation — single-field vs all-fields trajectory comparison (Exp A)
  7. plot_flow_field_correlation — flow level vs field preference analysis (Exp B)
  8. Entry-point functions for both irregular (GAT_ODE_irregular) and regular (DCRNN) models
"""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib.patches import Patch


# ==============================================================================
# 0. Global style configuration (publication-quality)
# ==============================================================================

# Paul Tol "Bright" — colourblind-friendly, print-safe
FIELD_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE']
FLOW_TRUE_COLOR = '#222255'
FLOW_PRED_COLOR = '#CC3311'
MULTI_FIELD_COLOR = FLOW_PRED_COLOR  # combined/all-fields prediction = prediction red
SINGLE_FIELD_LINESTYLE = '--'

PAPER_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.05,
}


def apply_paper_style():
    """Apply publication-quality rcParams. Call at the start of each plot function."""
    for k, v in PAPER_STYLE.items():
        plt.rcParams[k] = v


def _spine_cleanup(ax):
    """Remove top/right spines and set tick direction."""
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.6)


# ==============================================================================
# 1. Core trajectory recording
# ==============================================================================

@torch.no_grad()
def record_ode_trajectory_euler(ode_func, init_net, z0_flat, t_eval, num_fields):
    """
    Manually integrate the ODE step-by-step using Euler method, recording all
    intermediate states (field weights and individual vector field outputs).

    Args:
        ode_func:   ODEFunc instance with .funcs (ModuleList) and .logits_net
        init_net:   nn.Module mapping z -> initial logits
        z0_flat:    [BN, D] initial hidden state
        t_eval:     [T,] sorted time grid
        num_fields:  int, number of basis fields

    Returns:
        dict with keys:
            't_grid':    [T]              time points
            'weights':   [T, BN, M]       softmax weights at each step
            'f_outputs': [T, BN, M, D]    individual f_i(z) at each step
            'z_traj':    [T, BN, D]       hidden state trajectory
            'dz_traj':   [T, BN, D]       combined dz/dt at each step
    """
    device = z0_flat.device
    BN, D = z0_flat.shape
    T = len(t_eval)

    w0 = init_net(z0_flat)                    # [BN, M]
    state = (z0_flat.clone(), w0.clone())

    weights_list = []
    f_outputs_list = []
    z_list = []
    dz_list = []

    for i in range(T):
        z, logits = state
        w = F.softmax(logits, dim=-1)         # [BN, M]

        f_vals = torch.stack([f(z) for f in ode_func.funcs], dim=1)  # [BN, M, D]
        dz_combined = (w.unsqueeze(-1) * f_vals).sum(dim=1)           # [BN, D]

        weights_list.append(w.cpu())
        f_outputs_list.append(f_vals.cpu())
        z_list.append(z.cpu())
        dz_list.append(dz_combined.cpu())

        if i < T - 1:
            dt = t_eval[i + 1] - t_eval[i]
            dz, dlogits = ode_func(t_eval[i], state)
            z_new = z + dt * dz
            logits_new = logits + dt * dlogits
            state = (z_new, logits_new)

    return {
        't_grid':    t_eval.cpu(),
        'weights':   torch.stack(weights_list, dim=0),    # [T, BN, M]
        'f_outputs': torch.stack(f_outputs_list, dim=0),  # [T, BN, M, D]
        'z_traj':    torch.stack(z_list, dim=0),          # [T, BN, D]
        'dz_traj':   torch.stack(dz_list, dim=0),         # [T, BN, D]
    }


# ==============================================================================
# 2. Single-field ODE integration (for ablation study — Experiment A)
# ==============================================================================

@torch.no_grad()
def run_single_field_integration(ode_func, readout, z0_flat, t_eval, field_idx):
    """
    Integrate ODE using only a single field's vector field f_i, bypassing
    the softmax combination and logits evolution. Applies the same soft norm
    clipping as the original ODEFunc.

    Args:
        ode_func:   ODEFunc instance with .funcs (ModuleList)
        readout:    nn.Module mapping z -> flow (1-dim output)
        z0_flat:    [BN, D] initial hidden state
        t_eval:     [T,] sorted time grid
        field_idx: int, which field to use (0-indexed)

    Returns:
        preds: [BN, T] flow predictions from this single field
    """
    device = z0_flat.device
    z = z0_flat.clone()
    T = len(t_eval)
    z_traj = [z.clone()]

    for i in range(T - 1):
        dt = t_eval[i + 1] - t_eval[i]
        dz = ode_func.funcs[field_idx](z)
        # --- soft norm clipping (same as ODEFunc.forward) ---
        dz_norm = torch.norm(dz, dim=-1, keepdim=True)
        dz = dz * torch.tanh(dz_norm) / dz_norm.clamp(min=1e-6)
        z = z + dt * dz
        z_traj.append(z.clone())

    z_traj = torch.stack(z_traj, dim=0)  # [T, BN, D]
    flow = readout(z_traj)               # [T, BN, 1]
    flow = flow.squeeze(-1)              # [T, BN]
    return flow.permute(1, 0)            # [BN, T]


# ==============================================================================
# 3. Visualization functions
# ==============================================================================

def plot_field_weights_vs_flow(record, flow_true, t_grid_pred,
                              node_idx=0, batch_idx=0, BN=None,
                              save_path=None):
    """
    Publication-quality dual-subplot figure:
      Top:    softmax weights of each ODE field over time
      Bottom: true traffic flow over time

    Args:
        record:       dict from record_ode_trajectory_euler
        flow_true:    [T_pred,] or [B,N,P,Tp] ground-truth flow (will be flattened)
        t_grid_pred:  [T_pred,] time grid for the flow (in relative hours)
        node_idx:     which node to visualize
        batch_idx:    which batch element
        BN:           total BN flat dimension
        save_path:    path to save figure
    """
    apply_paper_style()

    # --- Prepare ODE weight data ---
    weights = record['weights']          # [T_ode, BN_all, M]
    t_ode = record['t_grid'].numpy()     # [T_ode]
    M = weights.shape[-1]

    if BN is not None:
        flat_idx = batch_idx * (BN // max(1, weights.shape[1] // BN)) + node_idx
        flat_idx = min(flat_idx, weights.shape[1] - 1)
    else:
        flat_idx = node_idx

    w_np = weights[:, flat_idx, :].numpy()  # [T_ode, M]

    # --- Prepare flow data ---
    if isinstance(flow_true, torch.Tensor):
        ft = flow_true.detach().cpu().numpy()
    else:
        ft = np.array(flow_true)
    if isinstance(t_grid_pred, torch.Tensor):
        tp = t_grid_pred.detach().cpu().numpy()
    else:
        tp = np.array(t_grid_pred)

    ft_1d = ft.reshape(-1)
    tp_1d = tp.reshape(-1)

    valid_mask = ~(np.isnan(ft_1d) | np.isinf(ft_1d))
    if valid_mask.sum() > 0:
        tp_1d = tp_1d[valid_mask]
        ft_1d = ft_1d[valid_mask]

    sort_idx = np.argsort(tp_1d)
    tp_1d = tp_1d[sort_idx]
    ft_1d = ft_1d[sort_idx]

    # --- Create figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)

    # --- Top: Field weights ---
    for m in range(M):
        ax1.plot(t_ode, w_np[:, m], color=FIELD_COLORS[m], linewidth=1.6,
                 label=f'Field {m + 1}', alpha=0.88)
    ax1.set_ylabel('Field Weight', fontweight='bold')
    ax1.set_title(f'ODE Field Weights Over Time (Node {node_idx})', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=min(M, 3),
               framealpha=0.85, edgecolor='#cccccc')
    ax1.set_axisbelow(True)
    ax1.grid(True, color='#e0e0e0', linewidth=0.7)
    ax1.set_ylim(-0.05, 1.05)
    _spine_cleanup(ax1)

    # --- Bottom: True flow (single series — no legend needed) ---
    ax2.plot(tp_1d, ft_1d, 'o-', color=FLOW_TRUE_COLOR, markersize=3.5, linewidth=2.0,
             markeredgecolor='white', markeredgewidth=0.5, alpha=0.9)
    ax2.set_xlabel('Relative Time', fontweight='bold')
    ax2.set_ylabel('Traffic Flow', fontweight='bold')
    ax2.set_title(f'True Flow (Node {node_idx})', fontweight='bold')
    ax2.set_axisbelow(True)
    ax2.grid(True, color='#e0e0e0', linewidth=0.7)
    _spine_cleanup(ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


def plot_weight_flow_overlay(record, flow_pred, flow_true, t_grid_pred,
                              node_idx=0, batch_idx=0, BN=None,
                              save_path=None):
    """
    Publication-quality single-figure: flow curve with dominant field background bands.

    The background color at each time step indicates which field has the
    highest weight, showing the relationship between flow regime and
    dominant ODE field.

    Args:
        (same as plot_field_weights_vs_flow)
    """
    apply_paper_style()

    # --- Prepare ODE weight data ---
    weights = record['weights']          # [T_ode, BN_all, M]
    t_ode = record['t_grid'].numpy()     # [T_ode]
    M = weights.shape[-1]

    if BN is not None:
        flat_idx = batch_idx * (BN // max(1, weights.shape[1] // BN)) + node_idx
        flat_idx = min(flat_idx, weights.shape[1] - 1)
    else:
        flat_idx = node_idx

    w_np = weights[:, flat_idx, :].numpy()  # [T_ode, M]
    dominant_field = np.argmax(w_np, axis=1)  # [T_ode]

    # --- Prepare flow data ---
    if isinstance(flow_pred, torch.Tensor):
        fp = flow_pred.detach().cpu().numpy()
    else:
        fp = np.array(flow_pred)
    if isinstance(flow_true, torch.Tensor):
        ft = flow_true.detach().cpu().numpy()
    else:
        ft = np.array(flow_true)
    if isinstance(t_grid_pred, torch.Tensor):
        tp = t_grid_pred.detach().cpu().numpy()
    else:
        tp = np.array(t_grid_pred)

    fp_1d = fp.reshape(-1)
    ft_1d = ft.reshape(-1)
    tp_1d = tp.reshape(-1)

    valid_mask = (ft_1d != 0) | (fp_1d != 0)
    if valid_mask.sum() > 0:
        tp_1d = tp_1d[valid_mask]
        fp_1d = fp_1d[valid_mask]
        ft_1d = ft_1d[valid_mask]

    sort_idx = np.argsort(tp_1d)
    tp_1d = tp_1d[sort_idx]
    fp_1d = fp_1d[sort_idx]
    ft_1d = ft_1d[sort_idx]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Background bands for dominant field
    t_ode_plot = np.array(t_ode)
    for i in range(len(t_ode_plot) - 1):
        field = dominant_field[i]
        ax.axvspan(t_ode_plot[i], t_ode_plot[i + 1],
                    alpha=0.18, color=FIELD_COLORS[field], zorder=1, linewidth=0)

    # Flow curves
    ax.plot(tp_1d, ft_1d, 'o-', color=FLOW_TRUE_COLOR, markersize=4, linewidth=2.2,
            label='True Flow', zorder=3, markeredgewidth=0.5,
            markeredgecolor='white')
    ax.plot(tp_1d, fp_1d, 's--', color=FLOW_PRED_COLOR, markersize=4, linewidth=2.2,
            label='Predicted Flow', zorder=3, markeredgewidth=0.5,
            markeredgecolor='white')

    # Legend: fields + flow curves
    legend_elements = [Patch(facecolor=FIELD_COLORS[m], alpha=0.35,
                              label=f'Field {m + 1} dominant')
                       for m in range(M)]
    legend_elements += [
        plt.Line2D([0], [0], color=FLOW_TRUE_COLOR, marker='o', markersize=5,
                   linewidth=2, label='True Flow'),
        plt.Line2D([0], [0], color=FLOW_PRED_COLOR, marker='s', markersize=5,
                   linestyle='--', linewidth=2, label='Predicted Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
              ncol=2, framealpha=0.85, edgecolor='#cccccc')

    ax.set_xlabel('Relative Time (hours)', fontweight='bold')
    ax.set_ylabel('Traffic Flow', fontweight='bold')
    ax.set_title(f'Flow Prediction with Dominant ODE Field (Node {node_idx})',
                 fontweight='bold')
    ax.set_axisbelow(True)
    ax.grid(True, color='#e0e0e0', linewidth=0.7, zorder=0)
    _spine_cleanup(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


def plot_vector_field_tsne(record, save_path=None, max_samples_per_field=500,
                            perplexity=30):
    """
    Publication-quality t-SNE of vector field outputs f_i(z), colored by field.

    Well-separated clusters indicate that different fields learn
    characteristically different functions.

    Args:
        record:              dict from record_ode_trajectory_euler
        save_path:           path to save figure
        max_samples_per_field: max samples to use per field
        perplexity:          t-SNE perplexity
    """
    apply_paper_style()

    f_outputs = record['f_outputs']  # [T, BN, M, D]
    T, BN, M, D = f_outputs.shape

    total_samples = T * BN
    samples_per_field = min(max_samples_per_field, total_samples)

    f_flat = f_outputs.reshape(-1, M, D)  # [T*BN, M, D]
    n_total = f_flat.shape[0]

    indices = np.random.choice(n_total, size=min(samples_per_field, n_total),
                                replace=False)

    all_samples = []
    all_labels = []
    for m in range(M):
        samples_m = f_flat[indices, m, :].numpy()
        all_samples.append(samples_m)
        all_labels.append(np.full(samples_m.shape[0], m))

    data = np.concatenate(all_samples, axis=0)  # [M*S, D]
    labels = np.concatenate(all_labels, axis=0)

    if data.shape[0] < 5:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, 'Not enough samples for t-SNE', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _spine_cleanup(ax)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        return

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(perplexity, data.shape[0] - 1),
                learning_rate='auto', init='pca', metric='euclidean')
    emb_2d = tsne.fit_transform(data)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for m in range(M):
        mask_m = labels == m
        ax.scatter(emb_2d[mask_m, 0], emb_2d[mask_m, 1],
                   c=[FIELD_COLORS[m]], label=f'Field {m + 1}',
                   alpha=0.5, s=16, edgecolors='white', linewidth=0.4, zorder=2)

    ax.set_title('t-SNE of Vector Field Outputs  $f_i(z)$  by ODE Field',
                 fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, markerscale=2,
              framealpha=0.85, edgecolor='#cccccc')
    ax.set_axisbelow(True)
    ax.grid(True, color='#e0e0e0', linewidth=0.7)
    _spine_cleanup(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# 4. New experiments
# ==============================================================================

def plot_field_ablation(single_field_preds, multi_field_pred, flow_true,
                          t_grid, node_idx, num_fields, save_path=None):
    """
    Experiment A: Compare single-field trajectories against the combined
    all-fields prediction and ground truth on the same axes.

    Directly demonstrates WHY multiple basis fields are needed — each single field
    fits a different portion of the time horizon, and only their combination
    achieves good fit throughout.

    Args:
        single_field_preds: list of [K] numpy arrays, one per field
        multi_field_pred:   [K] numpy array
        flow_true:           [K] numpy array
        t_grid:              [K] numpy array
        node_idx:            int
        num_fields:           int
        save_path:           path to save figure
    """
    apply_paper_style()

    # NaN safety
    flow_true = np.nan_to_num(np.asarray(flow_true), nan=0.0)
    multi_field_pred = np.nan_to_num(np.asarray(multi_field_pred), nan=0.0)
    t_grid = np.asarray(t_grid)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Ground truth — thick solid navy line with circle markers (observed data)
    ax.plot(t_grid, flow_true, '-', color=FLOW_TRUE_COLOR, linewidth=2.6,
            marker='o', markersize=4.5, markeredgecolor='white', markeredgewidth=0.5,
            label='Ground Truth', zorder=6, alpha=0.95)

    # All fields combined — thick solid red line with square markers
    ax.plot(t_grid, multi_field_pred, '-', color=MULTI_FIELD_COLOR,
            linewidth=2.2, marker='s', markersize=4, markeredgecolor='white',
            markeredgewidth=0.5, label='All Fields (Combined)', zorder=5, alpha=0.9)

    # Individual fields — thinner dashed/dotted lines
    for m in range(num_fields):
        sp = np.nan_to_num(np.asarray(single_field_preds[m]), nan=0.0)
        linestyle = ['--', '-.', ':'][m % 3]
        ax.plot(t_grid, sp, linestyle, color=FIELD_COLORS[m], linewidth=1.3,
                label=f'Field {m + 1} only', zorder=3, alpha=0.75)

    ax.set_xlabel('Prediction Step', fontweight='bold')
    ax.set_ylabel('Traffic Flow', fontweight='bold')
    ax.set_title(f'ODE Field Ablation Study (Node {node_idx})',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=min(num_fields + 2, 3),
              framealpha=0.85, edgecolor='#cccccc')
    ax.set_axisbelow(True)
    ax.grid(True, color='#e0e0e0', linewidth=0.7)
    _spine_cleanup(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


def plot_flow_field_correlation(all_flow_true, all_field_weights, num_fields,
                                save_path=None, n_bins=4):
    """
    Experiment B: Statistical association between traffic flow level and
    field activation preference.

    Data is binned by flow level (quartiles). For each bin, the mean
    field weight is shown with error bars (±1 std).

    Interpretation:
      If different fields show systematically different weight distributions
      across flow levels (e.g., Field 1 dominates low-flow bins while Field 2
      dominates high-flow bins), this indicates the model has learned to
      associate distinct dynamical regimes with different flow magnitudes —
      without being explicitly told to do so.

    The dashed horizontal line marks the uniform weight 1/M — the expected
    value if fields showed no specialization at all.

    Args:
        all_flow_true:     [N_total] numpy array of flow values
        all_field_weights:  [N_total, M] numpy array of field weights
        num_fields:         int
        save_path:         path to save figure
        n_bins:            number of flow bins (default 4 = quartiles)
    """
    apply_paper_style()

    flow = np.asarray(all_flow_true).reshape(-1)
    weights = np.asarray(all_field_weights).reshape(-1, num_fields)

    # Remove NaN/Inf entries
    valid = ~(np.isnan(flow) | np.isinf(flow))
    valid &= ~np.any(np.isnan(weights) | np.isinf(weights), axis=1)
    flow = flow[valid]
    weights = weights[valid]

    if len(flow) < n_bins * 5:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f'Insufficient data ({len(flow)} points) '
                f'for flow-field correlation', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        _spine_cleanup(ax)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
        return

    # Bin by flow quantiles
    q_edges = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(flow, q_edges)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    means = np.zeros((n_bins, num_fields))
    stds = np.zeros((n_bins, num_fields))
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (flow >= bin_edges[b]) & (flow < bin_edges[b + 1])
        counts[b] = mask.sum()
        if counts[b] > 0:
            means[b] = weights[mask].mean(axis=0)
            stds[b] = weights[mask].std(axis=0)

    # --- Plot: grouped bar chart, publication style ---
    fig, ax = plt.subplots(figsize=(8.5, 5))

    x = np.arange(n_bins)
    total_width = 0.7
    bar_width = total_width / num_fields

    for m in range(num_fields):
        offset = (m - (num_fields - 1) / 2) * bar_width
        ax.bar(x + offset, means[:, m], bar_width * 0.82,
               yerr=stds[:, m],
               color=FIELD_COLORS[m], label=f'Field {m + 1}',
               edgecolor='white', linewidth=0.8,
               error_kw={'elinewidth': 1.1, 'capsize': 2.5, 'capthick': 1.1,
                         'ecolor': '#666666'},
               zorder=3)

    # Uniform-weight reference line (no-specialization baseline); explained
    # in the legend instead of a floating annotation to avoid label clutter.
    uniform = 1.0 / num_fields
    ax.axhline(y=uniform, color='#888888', linewidth=1.0,
               linestyle=(0, (5, 3)), zorder=1)

    # One two-line tick label per group (name + count / flow range) kept in a
    # single text object so the lines can never collide with each other.
    bin_names = ['Low', 'Med-Low', 'Med-High', 'High']
    tick_labels = []
    for i in range(n_bins):
        lo = np.percentile(flow, q_edges[i])
        hi = np.percentile(flow, q_edges[i + 1])
        name = bin_names[i] if i < len(bin_names) else f'Q{i + 1}'
        tick_labels.append(f'{name}  (n={counts[i]})\n[{lo:.0f}, {hi:.0f}]')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5)

    ax.set_xlabel('Traffic Flow Level', fontweight='bold')
    ax.set_ylabel('Mean Field Weight', fontweight='bold')
    ax.set_title('Field Activation Distribution Across Traffic Flow Levels',
                 fontweight='bold', fontsize=13)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color='#888888', linewidth=1.0,
                              linestyle=(0, (5, 3)),
                              label=f'Uniform baseline ({uniform:.2f})'))
    ax.legend(handles=handles, loc='upper right', fontsize=8.5,
              framealpha=0.9, edgecolor='#cccccc', ncol=min(num_fields + 1, 4))

    ax.set_ylim(0, max((means + stds).max() * 1.18, uniform * 1.5))
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color='#e0e0e0', linewidth=0.7, zorder=0)
    _spine_cleanup(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()


# ==============================================================================
# 5. Entry-point functions for each model type
# ==============================================================================

@torch.no_grad()
def save_ode_visualizations_stpbgat(model, dataloader, cfg, save_dir, epoch,
                                     max_batches=1, max_nodes=5, seed=None):
    """
    Generate ODE trajectory visualizations for the irregular-sampling model.

    Produces:
      - weights_flow.pdf       per-node  (existing, polished)
      - overlay.pdf            per-node  (existing, polished)
      - tsne.pdf               global    (existing, polished)
      - ablation.pdf           per-node  (NEW — Experiment A)
      - flow_field_correlation.pdf global (NEW — Experiment B)

    Args:
        model:      GAT_ODE_irregular instance
        dataloader: DataLoader yielding {'features': ..., 'obs_mask': ..., 'pred_mask': ...}
        cfg:        Config object
        save_dir:   directory to save plots
        epoch:      current epoch (for naming)
        max_batches: how many batches to process (randomly selected)
        max_nodes:  how many nodes to visualize per batch
        seed:       random seed for batch/node selection (None = random each run)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    from lib.utils import StandardScaler
    standard_scaler_flow = StandardScaler(mean=cfg.MEAN_FLOW, std=cfg.STD_FLOW, device=cfg.DEVICE)
    standard_scaler_period = StandardScaler(mean=cfg.MEAN_PERIOD, std=cfg.STD_PERIOD,
                                             device=cfg.DEVICE)

    num_fields = model.ode_decoder.num_modes
    if num_fields <= 1:
        print(f"[ODE Viz] Only {num_fields} field(s), skipping visualization")
        return

    # Accumulate for flow-field correlation (Experiment B)
    all_flow_true_list = []
    all_field_weights_list = []

    # Randomly select which batches to visualize (test loader is unshuffled)
    rng = np.random.default_rng(seed)
    n_batches = len(dataloader)
    batch_indices = set(int(i) for i in rng.choice(
        n_batches, size=min(max_batches, n_batches), replace=False))
    print(f"[ODE Viz] Randomly selected batches: {sorted(batch_indices)}")

    batches_done = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx not in batch_indices:
            continue
        if batches_done >= max_batches:
            break

        features = batch['features'].to(cfg.DEVICE)
        obs_mask = batch['obs_mask'].to(cfg.DEVICE)
        pred_mask = batch['pred_mask'].to(cfg.DEVICE)

        x = features[:, :, :cfg.OBS_PATCH_LEN, :, :]
        y = features[:, :, cfg.OBS_PATCH_LEN:, :, :]
        x_mask = obs_mask[:, :, :cfg.OBS_PATCH_LEN, :]
        y_mask = pred_mask[:, :, cfg.OBS_PATCH_LEN:, :]

        # Normalize
        x_flow_norm = standard_scaler_flow.transform(x[..., -1].clone())
        x_period_norm = standard_scaler_period.transform(x[..., -2].clone())
        x_norm = x.clone()
        x_norm = x_norm.float()
        x_norm[..., -1] = x_flow_norm
        x_norm[..., -2] = x_period_norm

        y_norm = y.clone().float()
        y_norm[..., -2] = standard_scaler_period.transform(y[..., -2].clone())

        B, N, P_pred, Tp, F_feat = y.shape

        # --- Run encoder manually to get z0_flat ---
        b, n, p_obs, pl, _ = x_norm.shape
        x_value = x_norm[..., 4]
        patch_valid = (x_mask.sum(dim=-1) > 0)

        x_tem_emb = model.time_embedding(x_norm[..., 1:3], x_mask)
        x_period_emb = model.period_embedding(x_norm[..., 3].unsqueeze(-1)) * x_mask.unsqueeze(-1)
        x_value_emb = model.flow_embedding(x_value.unsqueeze(-1)) * x_mask.unsqueeze(-1)
        x_emb = x_tem_emb + x_period_emb + x_value_emb

        edges_mean, edges_sharp = model.ttcn(x_emb, x_mask, x_value)

        nodeA_mean = model.nodeA_proj_mean(model.nodeA_emb_mean).unsqueeze(0).expand(b, -1, -1)
        nodeA_sharp = model.nodeA_proj_sharp(model.nodeA_emb_sharp).unsqueeze(0).expand(b, -1, -1)
        nodeB_mean = model.nodeB_proj_mean(model.nodeB_emb_mean).unsqueeze(0).expand(b, -1, -1)
        nodeB_sharp = model.nodeB_proj_sharp(model.nodeB_emb_sharp).unsqueeze(0).expand(b, -1, -1)

        for i in range(model.n_layers):
            mask_for_layer = patch_valid if i == 0 else None
            new_edges, new_nodeA, new_nodeB, _, _ = model.graph_layers_mean[i](
                edges_mean, nodeA_mean, nodeB_mean, mask_for_layer)
            edges_mean = edges_mean + new_edges
            nodeA_mean = nodeA_mean + new_nodeA
            nodeB_mean = nodeB_mean + new_nodeB
        edges_mean = model.edges_norm_mean(edges_mean)

        for i in range(model.n_layers):
            mask_for_layer = patch_valid if i == 0 else None
            new_edges, new_nodeA, new_nodeB, _, _ = model.graph_layers_sharp[i](
                edges_sharp, nodeA_sharp, nodeB_sharp, mask_for_layer)
            edges_sharp = edges_sharp + new_edges
            nodeA_sharp = nodeA_sharp + new_nodeA
            nodeB_sharp = nodeB_sharp + new_nodeB
        edges_sharp = model.edges_norm_sharp(edges_sharp)

        edges = model.fusion_layer(edges_mean, edges_sharp)

        patch_valid_obs = patch_valid.float()
        valid_count = patch_valid_obs.sum(dim=-1, keepdim=True).clamp(min=1.0)
        z0 = (edges * patch_valid_obs.unsqueeze(-1)).sum(dim=2) / valid_count
        z0_flat = z0.reshape(b * n, model.dim)

        # --- Build time grid ---
        y_timestamps = y[..., 0].clone()
        y_mask_bool = y_mask.bool()
        y_ts_masked = y_timestamps.clone()
        y_ts_masked[~y_mask_bool] = float('inf')
        ref_t, _ = y_ts_masked.view(b, n, -1).min(dim=-1)
        ref_t = torch.where(torch.isinf(ref_t), torch.zeros_like(ref_t), ref_t)

        rel_ts = (y_timestamps - ref_t.unsqueeze(-1).unsqueeze(-1)) / model.time_scale
        rel_ts = rel_ts.float()
        rel_ts_valid = rel_ts[y_mask_bool]
        if rel_ts_valid.numel() == 0:
            batches_done += 1
            continue

        rel_ts_rounded = torch.round(rel_ts_valid * 1e4) / 1e4
        t_grid = torch.unique(torch.sort(rel_ts_rounded)[0])
        if t_grid[0].item() > 1e-8:
            t_grid = torch.cat([torch.zeros(1, device=z0.device, dtype=t_grid.dtype), t_grid])

        # --- Record ODE trajectory ---
        record = record_ode_trajectory_euler(
            model.ode_decoder.ode_func,
            model.ode_decoder.mode_net,
            z0_flat, t_grid, num_fields
        )

        # --- Get predictions ---
        pred_flat = model.ode_decoder(z0_flat, t_grid)
        T_grid = len(t_grid)
        pos = torch.searchsorted(t_grid, rel_ts).clamp(0, T_grid - 1)
        pred_flat_1d = pred_flat.reshape(-1)
        flat_idx = (torch.arange(b * n, device=z0.device).view(b, n, 1, 1) * T_grid + pos)
        output = pred_flat_1d[flat_idx]
        output = output * y_mask.float()
        output = standard_scaler_flow.inverse_transform(output)

        true_flow = y[..., -1].squeeze(-1)

        # --- Select nodes with sufficient valid data ---
        valid_counts = y_mask_bool[0].sum(dim=(1, 2))
        MIN_VALID = 5
        eligible = torch.where(valid_counts >= MIN_VALID)[0]
        if len(eligible) == 0:
            print(f"[ODE Viz] WARNING: No nodes with >= {MIN_VALID} valid points, skipping batch")
            batches_done += 1
            continue

        n_pick = min(max_nodes, len(eligible))
        picked = rng.choice(len(eligible), size=n_pick, replace=False)
        node_indices = [int(eligible[i]) for i in picked]

        first_node_done = False  # only do ablation for first viable node

        for ni in node_indices:
            flow_pred_ni = output[0, ni]        # [P_pred, Tp]
            flow_true_ni = true_flow[0, ni]      # [P_pred, Tp]
            rel_ts_ni = rel_ts[0, ni]            # [P_pred, Tp]
            y_mask_ni = y_mask_bool[0, ni]       # [P_pred, Tp]

            fp = flow_pred_ni[y_mask_ni].cpu()
            ft = flow_true_ni[y_mask_ni].cpu()
            tp = rel_ts_ni[y_mask_ni].cpu()

            n_valid = y_mask_ni.sum().item()
            if n_valid < 2:
                continue

            if torch.isnan(fp).any() or torch.isnan(ft).any():
                fp = torch.nan_to_num(fp, nan=0.0)
                ft = torch.nan_to_num(ft, nan=0.0)

            # --- Existing: Weight vs flow plot ---
            plot_field_weights_vs_flow(
                record, ft, tp,
                node_idx=ni, BN=n,
                save_path=os.path.join(save_dir,
                                       f'epoch{epoch}_node{ni}_weights_flow.pdf')
            )

            # --- Existing: Weight-flow overlay ---
            plot_weight_flow_overlay(
                record, fp, ft, tp,
                node_idx=ni, BN=n,
                save_path=os.path.join(save_dir,
                                       f'epoch{epoch}_node{ni}_overlay.pdf')
            )

            # --- NEW Experiment A: Field ablation ---
            if not first_node_done:
                # Run single-field integrations
                single_preds = []
                for m in range(num_fields):
                    pred_single = run_single_field_integration(
                        model.ode_decoder.ode_func,
                        model.ode_decoder.readout,
                        z0_flat, t_grid, field_idx=m
                    )
                    # Map grid predictions back to observation times for this node
                    pred_single_ni = pred_single[ni]          # [T_grid]
                    pred_single_valid = pred_single_ni[pos[0, ni][y_mask_ni]]  # [K]
                    # Inverse-transform single-field prediction (same scale as gt)
                    pred_single_raw = standard_scaler_flow.inverse_transform(
                        pred_single_valid.view(1, 1, 1, -1))[0, 0, 0, :]
                    single_preds.append(pred_single_raw.cpu().numpy())

                # Combined all-fields prediction at observation times
                multi_ni = pred_flat[ni]                      # [T_grid]
                multi_valid = multi_ni[pos[0, ni][y_mask_ni]]  # [K]
                # Inverse-transform combined prediction
                multi_valid_raw = standard_scaler_flow.inverse_transform(
                    multi_valid.view(1, 1, 1, -1))[0, 0, 0, :].cpu().numpy()

                plot_field_ablation(
                    single_preds, multi_valid_raw, ft.numpy(), tp.numpy(),
                    node_idx=ni, num_fields=num_fields,
                    save_path=os.path.join(save_dir,
                                           f'epoch{epoch}_node{ni}_ablation.pdf')
                )
                first_node_done = True

            # --- Accumulate data for Experiment B ---
            pos_ni = pos[0, ni]                         # [P_pred, Tp]
            pos_valid = pos_ni[y_mask_ni]               # [K]
            flat_ni_weights = record['weights'][pos_valid, ni, :]  # [K, M]
            all_flow_true_list.append(ft.cpu().numpy())
            all_field_weights_list.append(flat_ni_weights.cpu().numpy())

        # t-SNE (global, across all nodes)
        plot_vector_field_tsne(
            record,
            save_path=os.path.join(save_dir, f'epoch{epoch}_tsne.pdf')
        )

        batches_done += 1

    # --- NEW Experiment B: Flow-field correlation (across all batches) ---
    if all_flow_true_list:
        all_flow_true_cat = np.concatenate(all_flow_true_list, axis=0)
        all_weights_cat = np.concatenate(all_field_weights_list, axis=0)
        plot_flow_field_correlation(
            all_flow_true_cat, all_weights_cat, num_fields,
            save_path=os.path.join(save_dir, f'epoch{epoch}_flow_field_correlation.pdf')
        )

    print(f"[ODE Viz] Saved visualizations to {save_dir}")


@torch.no_grad()
def save_ode_visualizations_dcrnn(model, dataloader, scaler, save_dir, epoch,
                                   max_batches=1, max_nodes=5, seed=None):
    """
    Generate ODE trajectory visualizations for the regular-sampling DCRNN model.

    Produces:
      - weights_flow.pdf       per-node  (existing, polished)
      - overlay.pdf            per-node  (existing, polished)
      - tsne.pdf               global    (existing, polished)
      - ablation.pdf           per-node  (NEW — Experiment A)
      - flow_field_correlation.pdf global (NEW — Experiment B)

    Args:
        model:      DCRNN instance
        dataloader: DataLoader yielding (inputs, targets) or dict
        scaler:     Scaler instance for normalization
        save_dir:   directory to save plots
        epoch:      current epoch
        max_batches: how many batches to process (randomly selected)
        max_nodes:  how many nodes to visualize per batch
        seed:       random seed for batch/node selection (None = random each run)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    num_fields = model.decoder.num_modes
    if num_fields <= 1:
        print(f"[ODE Viz] Only {num_fields} field(s), skipping visualization")
        return

    # Accumulate for flow-field correlation (Experiment B)
    all_flow_true_list = []
    all_field_weights_list = []

    # Randomly select which batches to visualize (test loader is unshuffled)
    rng = np.random.default_rng(seed)
    n_batches = len(dataloader)
    batch_indices = set(int(i) for i in rng.choice(
        n_batches, size=min(max_batches, n_batches), replace=False))
    print(f"[ODE Viz] Randomly selected batches: {sorted(batch_indices)}")

    batches_done = 0
    for batch_idx, data in enumerate(dataloader):
        if batch_idx not in batch_indices:
            continue
        if batches_done >= max_batches:
            break

        # --- Parse data format ---
        if isinstance(data, (list, tuple)):
            history_data, future_data = data[0], data[1] if len(data) > 1 else None
        elif isinstance(data, dict):
            history_data = data.get('inputs', data.get('history_data'))
            future_data = data.get('target', data.get('future_data'))
        else:
            history_data, future_data = data, None

        if history_data is None:
            batches_done += 1
            continue

        history_data = history_data.to(next(model.parameters()).device)
        if future_data is not None:
            future_data = future_data.to(next(model.parameters()).device)

        # --- Raw flow for ground truth ---
        if future_data is not None:
            flow_true_raw = future_data[..., 0].clone()
        else:
            flow_true_raw = None

        # --- Normalize ---
        if scaler is not None:
            history_data = scaler.transform(history_data)
            if future_data is not None:
                future_data = scaler.transform(future_data)

        B, L, N, C_in = history_data.shape
        H = model.horizon
        D = model.dim

        # --- Forward: encoder -> decoder ---
        z0 = model.encoder(history_data)                           # [B, N, D]
        z0_flat = z0.reshape(B * N, D)

        t_pred = torch.linspace(0, 1, H, device=z0.device)

        record = record_ode_trajectory_euler(
            model.decoder.ode_func,
            model.decoder.mode_net,
            z0_flat, t_pred, num_fields
        )

        # Combined all-fields prediction (normalized space)
        flow_norm = model.decoder(z0, t_pred)                      # [H, B, N, 1]
        flow_pred_norm = flow_norm.squeeze(-1).permute(1, 2, 0)   # [B, N, H]

        # --- Inverse normalize predictions ---
        if scaler is not None:
            flow_pred_full = torch.zeros(B, H, N, C_in, device=flow_pred_norm.device)
            flow_pred_full[..., 0] = flow_pred_norm.permute(0, 2, 1)
            flow_pred_full = scaler.inverse_transform(flow_pred_full)
            flow_pred = flow_pred_full[..., 0].permute(0, 2, 1)    # [B, N, H]
        else:
            flow_pred = flow_pred_norm

        # Ground truth (original space)
        if flow_true_raw is not None:
            flow_true = flow_true_raw.permute(0, 2, 1)             # [B, N, H]
        else:
            flow_true = torch.zeros_like(flow_pred)

        # --- Per-node visualizations ---
        node_indices = rng.choice(N, size=min(max_nodes, N), replace=False)
        first_node_done = False

        for ni in node_indices:
            ni = int(ni) if hasattr(ni, 'item') else ni
            fp = flow_pred[0, ni].cpu()       # [H]
            ft = flow_true[0, ni].cpu()       # [H]
            tp = t_pred.cpu()                 # [H]

            # --- Existing plots ---
            plot_field_weights_vs_flow(
                record, ft, tp,
                node_idx=ni, BN=N,
                save_path=os.path.join(save_dir,
                                       f'epoch{epoch}_node{ni}_weights_flow.pdf')
            )

            plot_weight_flow_overlay(
                record, fp, ft, tp,
                node_idx=ni, BN=N,
                save_path=os.path.join(save_dir,
                                       f'epoch{epoch}_node{ni}_overlay.pdf')
            )

            # --- NEW Experiment A: Field ablation ---
            if not first_node_done:
                single_preds = []
                for m in range(num_fields):
                    pred_single_flat = run_single_field_integration(
                        model.decoder.ode_func,
                        model.decoder.readout,
                        z0_flat, t_pred, field_idx=m
                    )  # [BN, H]
                    pred_single_ni = pred_single_flat[ni]  # [H]

                    # Inverse normalize single-field prediction
                    if scaler is not None:
                        pred_single_full = torch.zeros(1, H, 1, C_in, device=pred_single_ni.device)
                        pred_single_full[..., 0] = pred_single_ni.unsqueeze(0).unsqueeze(-1)
                        pred_single_full = scaler.inverse_transform(pred_single_full)
                        pred_single_ni_raw = pred_single_full[..., 0].squeeze()
                    else:
                        pred_single_ni_raw = pred_single_ni

                    single_preds.append(pred_single_ni_raw.cpu().numpy())

                plot_field_ablation(
                    single_preds, fp.numpy(), ft.numpy(), tp.numpy(),
                    node_idx=ni, num_fields=num_fields,
                    save_path=os.path.join(save_dir,
                                           f'epoch{epoch}_node{ni}_ablation.pdf')
                )
                first_node_done = True

            # --- Accumulate for Experiment B ---
            w_ni = record['weights'][:, ni, :]  # [H, M]
            all_flow_true_list.append(ft.cpu().numpy())
            all_field_weights_list.append(w_ni.cpu().numpy())

        # t-SNE
        plot_vector_field_tsne(
            record,
            save_path=os.path.join(save_dir, f'epoch{epoch}_tsne.pdf')
        )

        batches_done += 1

    # --- NEW Experiment B: Flow-field correlation ---
    if all_flow_true_list:
        all_flow_true_cat = np.concatenate(all_flow_true_list, axis=0)
        all_weights_cat = np.concatenate(all_field_weights_list, axis=0)
        plot_flow_field_correlation(
            all_flow_true_cat, all_weights_cat, num_fields,
            save_path=os.path.join(save_dir, f'epoch{epoch}_flow_field_correlation.pdf')
        )

    print(f"[ODE Viz] Saved visualizations to {save_dir}")
