"""
Visualise Mamba internal gating signals for input sequences.

Extracts and plots:
  - Δ (delta): the discretisation step / update gate
  - z gate: multiplicative output gate
  - B norm: input-to-state projection magnitude
  - C norm: state-to-output projection magnitude

Usage
-----
    python visualize_gates.py --checkpoint model_next.pt
    python visualize_gates.py --checkpoint model_next.pt --layer 1
    python visualize_gates.py --checkpoint model_next.pt --n-samples 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import umap

from masked_model_gpu_mamba_next import CausalTimeSeriesMamba
from dataset import SyntheticSongDataset


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']

    model = CausalTimeSeriesMamba(
        feature_dim=args.get('feature_dim', 20),
        d_model=args['d_model'],
        n_layers=args['n_layers'],
        d_state=args['d_state'],
        d_conv=args.get('d_conv', 4),
        expand=args.get('expand', 2),
        d_ff=args.get('d_ff', 512),
        dropout=0.0,
        max_horizon=args.get('max_horizon', 1),
    ).to(device)

    sd = CausalTimeSeriesMamba.migrate_state_dict(ckpt['model_state_dict'])
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        cleaned = CausalTimeSeriesMamba.migrate_state_dict(cleaned)
        model.load_state_dict(cleaned)

    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val MSE {ckpt.get('val_loss', '?'):.4f})")
    return model, args


@torch.no_grad()
def extract_gates(model, x, device, amp_dtype=torch.bfloat16):
    """Extract internal Mamba gating signals for a single sequence.

    Parameters
    ----------
    model : CausalTimeSeriesMamba
    x : (1, seq_len, feature_dim) tensor

    Returns
    -------
    gates : list of dicts, one per layer.  Each dict contains:
        'delta' : (seq_len, d_inner) — update gate (after softplus)
        'z_gate': (seq_len, d_inner) — output gate (after SiLU)
        'B_norm': (seq_len,) — L2 norm of B per time step
        'C_norm': (seq_len,) — L2 norm of C per time step
    """
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    x = x.to(device)

    # Run input projection to get h
    h = raw_model.input_proj(x)

    gates = []
    for layer in raw_model.layers:
        mamba = layer.mamba
        h_normed = layer.norm(h)

        # Replicate Mamba's internal projection steps
        xz = mamba.in_proj(h_normed)  # (1, L, 2*d_inner)
        d_inner = mamba.d_inner
        x_branch = xz[:, :, :d_inner]     # (1, L, d_inner)
        z_branch = xz[:, :, d_inner:]     # (1, L, d_inner)

        # Conv1d (causal, needs channel-first format)
        L = x_branch.shape[1]
        x_conv = x_branch.transpose(1, 2)  # (1, d_inner, L)
        x_conv = mamba.conv1d(x_conv)[:, :, :L]  # trim to original length
        x_conv = x_conv.transpose(1, 2)  # (1, L, d_inner)
        x_conv = mamba.act(x_conv)  # SiLU

        # x_proj → dt_raw, B, C
        x_dbl = mamba.x_proj(x_conv)  # (1, L, dt_rank + 2*d_state)
        dt_rank = mamba.dt_rank
        d_state = mamba.d_state
        dt_raw = x_dbl[:, :, :dt_rank]
        B = x_dbl[:, :, dt_rank:dt_rank + d_state]
        C = x_dbl[:, :, dt_rank + d_state:]

        # dt_proj + softplus → Δ
        delta = F.softplus(mamba.dt_proj(dt_raw))  # (1, L, d_inner)

        # z gate (SiLU applied during output gating)
        z_gate = F.silu(z_branch)  # (1, L, d_inner)

        gates.append({
            'delta': delta[0].float().cpu().numpy(),
            'z_gate': z_gate[0].float().cpu().numpy(),
            'B_norm': B[0].float().cpu().norm(dim=-1).numpy(),
            'C_norm': C[0].float().cpu().norm(dim=-1).numpy(),
        })

        # Advance h through the actual layer for next iteration
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            h = layer(h)

    return gates


def visualize_gates_single_layer(x_np, state_np, gates, layer_idx,
                                 save_path, max_channels=64):
    """Plot gating signals for one layer aligned with the input."""
    g = gates[layer_idx]
    seq_len = x_np.shape[0]

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(6, 2, height_ratios=[0.5, 2, 2, 2, 1, 1],
                          width_ratios=[1, 0.02], hspace=0.2, wspace=0.03)

    # State bar
    ax_state = fig.add_subplot(gs[0, 0])
    for t in range(seq_len):
        ax_state.axvspan(t, t + 1,
                         color=plt.cm.tab10(state_np[t] / 10),
                         alpha=0.8, linewidth=0)
    ax_state.set_xlim(0, seq_len)
    ax_state.set_yticks([])
    ax_state.set_ylabel('State', fontsize=9)
    ax_state.set_title(f'Mamba Layer {layer_idx + 1} — Internal Gates',
                       fontsize=13)
    plt.setp(ax_state.get_xticklabels(), visible=False)

    # Input signal
    vmax_x = max(abs(x_np.min()), abs(x_np.max()))
    ax_input = fig.add_subplot(gs[1, 0], sharex=ax_state)
    ax_input.imshow(x_np.T, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax_x, vmax=vmax_x,
                    extent=[0, seq_len, x_np.shape[1] - 0.5, -0.5],
                    interpolation='none')
    ax_input.set_ylabel('Input dim', fontsize=9)
    ax_input.set_title('Input signal', fontsize=10)
    plt.setp(ax_input.get_xticklabels(), visible=False)

    # Δ (delta) — update gate
    delta = g['delta'][:, :max_channels]
    ax_delta = fig.add_subplot(gs[2, 0], sharex=ax_state)
    im_d = ax_delta.imshow(delta.T, aspect='auto', cmap='magma',
                           extent=[0, seq_len, delta.shape[1] - 0.5, -0.5],
                           interpolation='none')
    ax_delta.set_ylabel('Channel', fontsize=9)
    ax_delta.set_title(f'Δ (update gate) — first {max_channels} channels  '
                       f'[mean={delta.mean():.3f}, max={delta.max():.3f}]',
                       fontsize=10)
    plt.setp(ax_delta.get_xticklabels(), visible=False)

    # z gate
    z = g['z_gate'][:, :max_channels]
    ax_z = fig.add_subplot(gs[3, 0], sharex=ax_state)
    ax_z.imshow(z.T, aspect='auto', cmap='viridis',
                extent=[0, seq_len, z.shape[1] - 0.5, -0.5],
                interpolation='none')
    ax_z.set_ylabel('Channel', fontsize=9)
    ax_z.set_title(f'z gate (output gate, SiLU) — first {max_channels} ch',
                   fontsize=10)
    plt.setp(ax_z.get_xticklabels(), visible=False)

    # B and C norms as line plots
    ax_bc = fig.add_subplot(gs[4, 0], sharex=ax_state)
    t = np.arange(seq_len)
    ax_bc.plot(t, g['B_norm'], alpha=0.8, linewidth=0.8, label='‖B‖')
    ax_bc.plot(t, g['C_norm'], alpha=0.8, linewidth=0.8, label='‖C‖')
    ax_bc.set_ylabel('Norm', fontsize=9)
    ax_bc.set_title('B (input→state) and C (state→output) norms', fontsize=10)
    ax_bc.legend(fontsize=8, loc='upper right')
    ax_bc.grid(True, alpha=0.2)
    plt.setp(ax_bc.get_xticklabels(), visible=False)

    # Δ mean across channels (summary)
    ax_dmean = fig.add_subplot(gs[5, 0], sharex=ax_state)
    delta_mean = g['delta'].mean(axis=1)
    ax_dmean.fill_between(t, 0, delta_mean, alpha=0.4, color='#d62728')
    ax_dmean.plot(t, delta_mean, color='#d62728', linewidth=0.8)
    ax_dmean.set_xlabel('Time step', fontsize=9)
    ax_dmean.set_ylabel('Mean Δ', fontsize=9)
    ax_dmean.set_title('Mean Δ across all channels (state update intensity)',
                       fontsize=10)
    ax_dmean.grid(True, alpha=0.2)

    # Colorbars
    ax_cb = fig.add_subplot(gs[2, 1])
    plt.colorbar(im_d, cax=ax_cb, label='Δ')
    for row in [0, 1, 3, 4, 5]:
        fig.add_subplot(gs[row, 1]).axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def visualize_gates_all_layers(x_np, state_np, gates, save_path):
    """Summary view: mean Δ for all layers in one plot."""
    n_layers = len(gates)
    seq_len = x_np.shape[0]

    fig = plt.figure(figsize=(18, 3 + 2 * n_layers))
    gs = fig.add_gridspec(2 + n_layers, 1,
                          height_ratios=[0.5, 2] + [1.5] * n_layers,
                          hspace=0.25)

    # State bar
    ax_state = fig.add_subplot(gs[0])
    for t in range(seq_len):
        ax_state.axvspan(t, t + 1,
                         color=plt.cm.tab10(state_np[t] / 10),
                         alpha=0.8, linewidth=0)
    ax_state.set_xlim(0, seq_len)
    ax_state.set_yticks([])
    ax_state.set_ylabel('State', fontsize=9)
    ax_state.set_title('Mamba Gate Dynamics — All Layers', fontsize=13)
    plt.setp(ax_state.get_xticklabels(), visible=False)

    # Input signal
    vmax_x = max(abs(x_np.min()), abs(x_np.max()))
    ax_input = fig.add_subplot(gs[1], sharex=ax_state)
    ax_input.imshow(x_np.T, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax_x, vmax=vmax_x,
                    extent=[0, seq_len, x_np.shape[1] - 0.5, -0.5],
                    interpolation='none')
    ax_input.set_ylabel('Input dim', fontsize=9)
    ax_input.set_title('Input signal', fontsize=10)
    plt.setp(ax_input.get_xticklabels(), visible=False)

    # Mean Δ for each layer
    t = np.arange(seq_len)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_layers))
    for i in range(n_layers):
        ax = fig.add_subplot(gs[2 + i], sharex=ax_state)
        delta_mean = gates[i]['delta'].mean(axis=1)
        ax.fill_between(t, 0, delta_mean, alpha=0.4, color=colors[i])
        ax.plot(t, delta_mean, color=colors[i], linewidth=0.8)
        ax.set_ylabel(f'L{i+1} Δ', fontsize=9)
        ax.grid(True, alpha=0.2)
        if i < n_layers - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Time step', fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def extract_gates_bulk(model, X, states, device, seq_len, batch_size=64,
                       amp_dtype=torch.bfloat16):
    """Extract gate vectors across the full dataset.

    Returns
    -------
    layer_deltas : list of (n_steps_used, d_inner) arrays
    layer_zgates : list of (n_steps_used, d_inner) arrays
    states_used  : (n_steps_used,) array
    """
    n_steps = X.shape[0]
    n_windows = n_steps // seq_len
    n_used = n_windows * seq_len
    X_trim = X[:n_used]

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    n_layers = len(raw_model.layers)

    all_deltas = [[] for _ in range(n_layers)]
    all_zgates = [[] for _ in range(n_layers)]
    all_B = [[] for _ in range(n_layers)]
    all_C = [[] for _ in range(n_layers)]
    all_xconv = [[] for _ in range(n_layers)]

    print(f"Extracting gates: {n_windows} windows × {seq_len} steps "
          f"= {n_used} / {n_steps} time steps")

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch_windows = []
        for w in range(start, end):
            t0 = w * seq_len
            batch_windows.append(X_trim[t0:t0 + seq_len])

        x_batch = torch.from_numpy(
            np.stack(batch_windows).astype(np.float32)
        ).to(device)

        h = raw_model.input_proj(x_batch)
        for li, layer in enumerate(raw_model.layers):
            mamba = layer.mamba
            h_normed = layer.norm(h)

            xz = mamba.in_proj(h_normed)
            d_inner = mamba.d_inner
            x_branch = xz[:, :, :d_inner]
            z_branch = xz[:, :, d_inner:]

            L = x_branch.shape[1]
            x_conv = x_branch.transpose(1, 2)
            x_conv = mamba.conv1d(x_conv)[:, :, :L].transpose(1, 2)
            x_conv = mamba.act(x_conv)

            x_dbl = mamba.x_proj(x_conv)
            dt_rank = mamba.dt_rank
            d_state = mamba.d_state
            dt_raw = x_dbl[:, :, :dt_rank]
            B = x_dbl[:, :, dt_rank:dt_rank + d_state]
            C = x_dbl[:, :, dt_rank + d_state:]

            delta = F.softplus(mamba.dt_proj(dt_raw))
            z_gate = F.silu(z_branch)

            def _to_np(t, d):
                return t.detach().float().cpu().numpy().reshape(-1, d)

            all_deltas[li].append(_to_np(delta, d_inner))
            all_zgates[li].append(_to_np(z_gate, d_inner))
            all_B[li].append(_to_np(B, d_state))
            all_C[li].append(_to_np(C, d_state))
            all_xconv[li].append(_to_np(x_conv, d_inner))

            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                h = layer(h)

    def _concat(lst):
        return [np.concatenate(c, axis=0) for c in lst]

    states_used = states[:n_used]
    return {
        'delta': _concat(all_deltas),
        'z_gate': _concat(all_zgates),
        'B': _concat(all_B),
        'C': _concat(all_C),
        'x_conv': _concat(all_xconv),
    }, states_used


def _umap_grid(layer_data, states_sub, n_circles, title, fname,
               label_prefix='Layer'):
    """Generic helper: UMAP grid of per-layer vectors, one panel per layer."""
    n_layers = len(layer_data)
    n_cols = min(n_layers, 4)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_layers):
        data = layer_data[i]
        dim = data.shape[1]
        print(f"Computing UMAP for {label_prefix} {i+1} ({dim}D)...")
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.3,
                            metric='euclidean', n_jobs=-1)
        emb = reducer.fit_transform(data)

        ax = axes[i]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=states_sub,
                        cmap='tab10', s=3, alpha=0.5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{label_prefix} {i+1} ({dim}D)', fontsize=12)
        ax.grid(True, alpha=0.2)

    cbar = plt.colorbar(sc, ax=axes[n_layers - 1], label='Circle index')
    cbar.set_ticks(range(n_circles))
    for ax_i in range(n_layers, len(axes)):
        axes[ax_i].axis('off')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


def visualize_gates_umap(gate_dict, states_used, n_circles,
                         umap_points=10000, save_dir='.'):
    """UMAP of all gate types per layer, colored by circle state."""
    rng = np.random.default_rng(42)
    n_total = len(states_used)
    if n_total > umap_points:
        idx = rng.choice(n_total, umap_points, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_total)

    states_sub = states_used[idx]
    save_dir = Path(save_dir)

    specs = [
        ('delta',  'Δ (update gate)', 'gates_umap_delta.png', 'Δ Layer'),
        ('z_gate', 'z (output gate)', 'gates_umap_zgate.png', 'z Layer'),
        ('B',      'B (input → state)', 'gates_umap_B.png', 'B Layer'),
        ('C',      'C (state → output)', 'gates_umap_C.png', 'C Layer'),
        ('x_conv', 'x after conv1d+SiLU (local features)',
         'gates_umap_xconv.png', 'x_conv Layer'),
    ]

    for key, title, fname, label in specs:
        layer_data = [ld[idx] for ld in gate_dict[key]]
        _umap_grid(layer_data, states_sub, n_circles,
                   f'UMAP of {title} by Layer',
                   save_dir / fname, label_prefix=label)


def main():
    parser = argparse.ArgumentParser(
        description='Visualise Mamba internal gating signals.')
    parser.add_argument('--checkpoint', type=str, default='model_next.pt')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Number of sequences to visualise')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer to plot in detail (1-based). '
                             'Default: plot all-layer summary.')
    parser.add_argument('--umap', action='store_true',
                        help='Run UMAP on gate vectors across the full '
                             'dataset, colored by circle state.')
    parser.add_argument('--umap-points', type=int, default=10000,
                        help='Max points for UMAP')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = (torch.bfloat16
                 if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                 else torch.float32)

    model, ckpt_args = load_model(args.checkpoint, device)
    seq_len = ckpt_args['seq_len']

    save_dir = Path(args.save_dir)

    if args.umap:
        # Full-dataset gate UMAP
        data_dir = Path(args.data_dir)
        npz = np.load(data_dir / 'data.npz')
        X = npz['X']
        states = npz['states']
        with open(data_dir / 'config.json') as f:
            config = json.load(f)
        print(f"Data: {X.shape[0]} steps × {X.shape[1]}D, "
              f"{config['n_circles']} circles")

        gate_dict, states_used = extract_gates_bulk(
            model, X, states, device, seq_len, amp_dtype=amp_dtype)

        visualize_gates_umap(gate_dict, states_used,
                             n_circles=config['n_circles'],
                             umap_points=args.umap_points,
                             save_dir=save_dir)
        return

    # Per-sample gate visualisation
    ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=seq_len,
        stride=seq_len // 2,
        mask_ratio=0.0,
        mask_patch_size=1,
        mask_seed=0,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n_samples, len(ds)),
                         replace=False)

    for i, idx in enumerate(indices):
        x, state, _ = ds[idx]
        x_in = x.unsqueeze(0).to(device)

        print(f"\nSample {i}: extracting gates for {seq_len} steps...")
        gates = extract_gates(model, x_in, device, amp_dtype)

        x_np = x.numpy()
        state_np = state.numpy()

        if args.layer is not None:
            layer_idx = args.layer - 1
            fname = save_dir / f'gates_layer{args.layer}_{i}.png'
            visualize_gates_single_layer(x_np, state_np, gates, layer_idx,
                                         fname)
        else:
            fname = save_dir / f'gates_summary_{i}.png'
            visualize_gates_all_layers(x_np, state_np, gates, fname)

            for li in [0, len(gates) - 1]:
                fname_detail = save_dir / f'gates_layer{li+1}_{i}.png'
                visualize_gates_single_layer(x_np, state_np, gates, li,
                                             fname_detail)


if __name__ == '__main__':
    main()
