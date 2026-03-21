"""
Evaluate learned representations from the Causal Mamba model.

Runs the full dataset through the trained model, extracts the
intermediate representation at each Mamba layer, and visualises
them with UMAP.  Optionally computes Levina-Bickel intrinsic
dimension estimates on each layer's representation.

Usage
-----
    python evaluate_representations.py                        # all layers
    python evaluate_representations.py --checkpoint best.pt   # custom ckpt
    python evaluate_representations.py --no-lb                # skip LB
    python evaluate_representations.py --layers 7             # layer 7 only
    python evaluate_representations.py --layers 1,4,7,output  # specific layers
    python evaluate_representations.py --layers input,7,output # with input
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
from sklearn.decomposition import PCA

from masked_model_gpu_mamba_next import CausalTimeSeriesMamba
from mamba_jepa import CausalMambaJEPA
from mamba_jepa_multiscale import CausalMambaJEPAMultiScale
from estimate_dimension import levina_bickel_estimator


def gpu_silhouette_score(X_np, labels, batch_size=2048):
    """GPU-accelerated silhouette score using batched pairwise distances."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(np.ascontiguousarray(X_np)).float().to(device)
    labels_t = torch.from_numpy(np.asarray(labels)).long().to(device)
    n = X.shape[0]
    unique_labels = torch.unique(labels_t)
    n_clusters = unique_labels.shape[0]

    if n_clusters < 2:
        return 0.0

    cluster_masks = labels_t.unsqueeze(0) == unique_labels.unsqueeze(1)
    cluster_sizes = cluster_masks.sum(dim=1).float()

    a_vals = torch.zeros(n, device=device)
    b_vals = torch.full((n,), float('inf'), device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(X[start:end], X)
        batch_labels = labels_t[start:end]

        for ci, cl in enumerate(unique_labels):
            mask_c = cluster_masks[ci]
            sum_dists = (dists * mask_c.unsqueeze(0).float()).sum(dim=1)
            count = cluster_sizes[ci]

            is_own = (batch_labels == cl)

            own_mean = sum_dists / (count - 1).clamp(min=1)
            a_vals[start:end] = torch.where(is_own, own_mean, a_vals[start:end])

            other_mean = sum_dists / count.clamp(min=1)
            cur_b = b_vals[start:end]
            b_vals[start:end] = torch.where(
                ~is_own & (other_mean < cur_b), other_mean, cur_b)

    s = (b_vals - a_vals) / torch.max(a_vals, b_vals).clamp(min=1e-10)
    return s.mean().item()


def gpu_knn(X_np, k, batch_size=2048):
    """Batched k-NN on GPU using PyTorch.

    Returns (distances, indices) each of shape (n, k).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(np.ascontiguousarray(X_np)).float().to(device)
    n = X.shape[0]
    all_dists = torch.zeros(n, k, device=device)
    all_idx = torch.zeros(n, k, dtype=torch.long, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(X[start:end], X)
        dists[:, start:end].fill_diagonal_(float('inf'))
        topk_d, topk_i = dists.topk(k, dim=1, largest=False)
        all_dists[start:end] = topk_d
        all_idx[start:end] = topk_i

    return all_dists.cpu().numpy(), all_idx.cpu().numpy()


def _fit_umap(panel_args):
    """Runs UMAP fit_transform for one panel, using precomputed k-NN if available."""
    title, data, lb_key, n_neighbors, pre = panel_args
    if pre is not None:
        knn_indices, knn_dists = pre
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.3,
            precomputed_knn=(knn_indices, knn_dists, None))
    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.3,
            metric='euclidean', n_jobs=-1)
    return reducer.fit_transform(data)


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint (supports CausalTimeSeriesMamba,
    CausalMambaJEPA, and CausalMambaJEPAMultiScale)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = ckpt['args']
    model_type = args.get('model_type', 'mamba_next')

    if model_type == 'mamba_jepa_multiscale':
        branch_strides = args.get('branch_strides', [4])
        layers_per_branch = args.get('layers_per_branch_parsed',
                                     [args.get('n_layers', 7)] * len(branch_strides))
        model = CausalMambaJEPAMultiScale(
            feature_dim=args.get('feature_dim', 20),
            d_model=args['d_model'],
            strides=branch_strides,
            layers_per_branch=layers_per_branch,
            d_state=args['d_state'],
            d_conv=args.get('d_conv', 4),
            expand=args.get('expand', 2),
            dropout=0.0,
            kernel_factor=args.get('kernel_factor', 2),
            predictor_n_layers=args.get('predictor_n_layers', 2),
            predictor_type='mlp' if args.get('predictor_mlp', False) else 'mamba',
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        val_metric = ckpt.get('val_loss', '?')
        strides_str = ','.join(str(s) for s in branch_strides)
        print(f"Loaded Multi-Scale Mamba-JEPA checkpoint from epoch {ckpt['epoch']}  "
              f"(val JEPA loss {val_metric:.4f}, strides=[{strides_str}])")
    elif model_type == 'mamba_jepa':
        model = CausalMambaJEPA(
            feature_dim=args.get('feature_dim', 20),
            d_model=args['d_model'],
            n_layers=args['n_layers'],
            d_state=args['d_state'],
            d_conv=args.get('d_conv', 4),
            expand=args.get('expand', 2),
            dropout=0.0,
            predictor_n_layers=args.get('predictor_n_layers', 2),
            predictor_type='mlp' if args.get('predictor_mlp', False) else 'mamba',
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        val_metric = ckpt.get('val_loss', '?')
        print(f"Loaded Mamba-JEPA checkpoint from epoch {ckpt['epoch']}  "
              f"(val JEPA loss {val_metric:.4f})")
    else:
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
        model.load_state_dict(sd)
        model.eval()
        print(f"Loaded Causal Mamba checkpoint from epoch {ckpt['epoch']}  "
              f"(val MSE {ckpt.get('val_loss', '?'):.4f})")

    return model, args


def extract_representations(model, X, device, batch_size=64, seq_len=512):
    """
    Run the full time series through the model and collect per-layer
    representations.

    Uses non-overlapping windows to cover the dataset, then flattens
    back to per-time-step representations.

    Returns
    -------
    layer_reps : list of ndarray, each (n_steps_used, d_model)
    states_used : ndarray (n_steps_used,)
        Aligned state labels.
    """
    n_steps = X.shape[0]
    # Trim to exact multiple of seq_len
    n_windows = n_steps // seq_len
    n_used = n_windows * seq_len
    X_trim = X[:n_used]

    print(f"Extracting representations: {n_windows} windows × {seq_len} steps "
          f"= {n_used} / {n_steps} time steps")

    # Process in batches
    # encode() returns n_encoder_layers + 1 (output projection)
    all_layers = None

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch_windows = []
        for w in range(start, end):
            t0 = w * seq_len
            batch_windows.append(X_trim[t0:t0 + seq_len])

        x_batch = torch.from_numpy(
            np.stack(batch_windows).astype(np.float32)
        ).to(device)

        layer_outputs = model.encode(x_batch)

        if all_layers is None:
            all_layers = [[] for _ in range(len(layer_outputs))]

        for i, lo in enumerate(layer_outputs):
            # lo: (batch, seq_len, dim) → flatten to (batch*seq_len, dim)
            all_layers[i].append(lo.cpu().numpy().reshape(-1, lo.shape[-1]))

    layer_reps = [np.concatenate(chunks, axis=0) for chunks in all_layers]
    return layer_reps, n_used


def extract_hidden_states(model, X, device, batch_size=64, seq_len=512):
    """Extract Mamba block hidden states (before residual connection).

    Uses forward hooks on each MambaLayer's internal ``mamba`` module to
    capture what the SSM itself outputs, without the skip connection.

    Returns
    -------
    hidden_reps : list of ndarray, each (n_steps_used, d_model)
        One entry per Mamba layer.
    n_used : int
    """
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    if hasattr(raw_model, 'layers'):
        mamba_layers = raw_model.layers
    elif hasattr(raw_model, 'target_encoder'):
        mamba_layers = raw_model.target_encoder.layers
    else:
        raise ValueError("Model does not have .layers or .target_encoder — "
                         "hidden state extraction not supported")

    n_steps = X.shape[0]
    n_windows = n_steps // seq_len
    n_used = n_windows * seq_len
    X_trim = X[:n_used]

    n_layers = len(mamba_layers)
    all_hidden = [[] for _ in range(n_layers)]

    # Storage for hook captures (reset each batch)
    captured = [None] * n_layers

    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output.detach()
        return hook

    # Register hooks on the mamba sub-module inside each MambaLayer
    hooks = []
    for i, layer in enumerate(mamba_layers):
        h = layer.mamba.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            for start in range(0, n_windows, batch_size):
                end = min(start + batch_size, n_windows)
                batch_windows = []
                for w in range(start, end):
                    t0 = w * seq_len
                    batch_windows.append(X_trim[t0:t0 + seq_len])

                x_batch = torch.from_numpy(
                    np.stack(batch_windows).astype(np.float32)
                ).to(device)

                _ = model.encode(x_batch)

                for i in range(n_layers):
                    all_hidden[i].append(
                        captured[i].cpu().numpy().reshape(-1, captured[i].shape[-1])
                    )
    finally:
        for h in hooks:
            h.remove()

    hidden_reps = [np.concatenate(chunks, axis=0) for chunks in all_hidden]
    print(f"Extracted {n_layers} Mamba hidden states: {hidden_reps[0].shape}")
    return hidden_reps, n_used


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BERT model intermediate representations.')
    parser.add_argument('--checkpoint', type=str, default='bert_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing data.npz')
    parser.add_argument('--umap-points', type=int, default=10000,
                        help='Max points for UMAP (subsampled if larger)')
    parser.add_argument('--umap-neighbors', type=int, default=50,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--no-lb', action='store_true',
                        help='Skip Levina-Bickel dimension estimates')
    parser.add_argument('--lb-points', type=int, default=2000,
                        help='Points to subsample for Levina-Bickel')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated list of layers to visualise. '
                             'Use integers for transformer layers (1-based), '
                             '"input" for raw input, "output" for output '
                             'projection.  E.g. --layers input,4,7,output. '
                             'Default: all layers.')
    parser.add_argument('--per-layer', action='store_true',
                        help='Save each layer as a separate PNG file '
                             '(umap_input.png, umap_layer_1.png, ...) '
                             'instead of a single grid.')
    parser.add_argument('--pca', type=int, default=0, metavar='N',
                        help='Apply PCA to reduce to N dimensions before '
                             'UMAP (0 = no PCA, run UMAP on raw data). '
                             'E.g. --pca 10 reduces 128D → 10D first.')
    parser.add_argument('--velocity', action='store_true',
                        help='Concatenate finite-difference velocity with '
                             'position before UMAP, doubling the '
                             'dimensionality.  Saves to '
                             'representation_umap_velocity.png.')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    # Load data
    data_dir = Path(args.data_dir)
    npz = np.load(data_dir / 'data.npz')
    X = npz['X']
    states = npz['states']
    with open(data_dir / 'config.json') as f:
        config = json.load(f)
    print(f"Data: {X.shape[0]} steps × {X.shape[1]}D, "
          f"{config['n_circles']} circles")

    # Load model
    model, model_args = load_model(args.checkpoint, device)
    seq_len = model_args['seq_len']

    # Extract representations
    layer_reps, n_used = extract_representations(
        model, X, device, batch_size=64, seq_len=seq_len,
    )
    states_used = states[:n_used]

    model_type = model_args.get('model_type', 'mamba_next')
    has_output_proj = model_type not in ('mamba_jepa', 'mamba_jepa_multiscale')

    if has_output_proj:
        n_encoder_layers = len(layer_reps) - 1
        print(f"Extracted {n_encoder_layers} encoder layers + output projection")
        print(f"  Encoder layers: {layer_reps[0].shape}")
        print(f"  Output projection: {layer_reps[-1].shape}")
    else:
        n_encoder_layers = len(layer_reps)
        print(f"Extracted {n_encoder_layers} encoder layers")
        print(f"  Encoder layers: {layer_reps[0].shape}")

    # Build the full list of available panels: input, layer_1..N, [output]
    all_panel_keys = ['input']
    for i in range(n_encoder_layers):
        all_panel_keys.append(f'layer_{i+1}')
    if has_output_proj:
        all_panel_keys.append('output')

    if args.layers is not None:
        selected_keys = []
        for token in args.layers.split(','):
            token = token.strip().lower()
            if token == 'input':
                selected_keys.append('input')
            elif token == 'output':
                selected_keys.append('output')
            else:
                try:
                    layer_num = int(token)
                    key = f'layer_{layer_num}'
                    if key not in all_panel_keys:
                        print(f"WARNING: layer {layer_num} does not exist "
                              f"(model has {n_encoder_layers} layers), skipping")
                    else:
                        selected_keys.append(key)
                except ValueError:
                    print(f"WARNING: unrecognised layer '{token}', skipping")
        if not selected_keys:
            print("No valid layers selected, falling back to all layers.")
            selected_keys = all_panel_keys
    else:
        selected_keys = all_panel_keys

    print(f"Visualising: {', '.join(selected_keys)}")

    # ---- Subsample for UMAP ----
    rng = np.random.default_rng(42)
    if n_used > args.umap_points:
        idx = rng.choice(n_used, args.umap_points, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_used)

    states_sub = states_used[idx]

    # ---- Velocity helper ----
    def _with_velocity(rep, n_used, seq_len):
        """Concatenate finite-difference velocity with position.

        Zeros out velocity at window boundaries where the difference
        would span two non-contiguous windows.
        """
        vel = np.zeros_like(rep)
        vel[:-1] = rep[1:] - rep[:-1]
        # Zero out boundary frames (last step of each window)
        for w in range(n_used // seq_len):
            vel[w * seq_len + seq_len - 1] = 0.0
        return np.concatenate([rep, vel], axis=1)

    # ---- Build panel data for selected layers ----
    def get_panel_data(key):
        """Return (title, data_subsampled, key) for a given panel key."""
        if key == 'input':
            data = X[:n_used].astype(np.float32)
            D = data.shape[1]
        elif key == 'output':
            data = layer_reps[-1]
            D = data.shape[1]
        else:
            layer_i = int(key.split('_')[1]) - 1
            data = layer_reps[layer_i]
            D = data.shape[1]

        if args.velocity:
            data = _with_velocity(data, n_used, seq_len)
            label = f'{D}+{D}D'
        else:
            label = f'{D}D'

        if key == 'input':
            title = f'Input ({label})'
        elif key == 'output':
            title = f'Output ({label})'
        else:
            layer_i = int(key.split('_')[1]) - 1
            title = f'Layer {layer_i+1} ({label})'

        return (title, data[idx], key)

    def get_lb_data(key):
        """Return data for Levina-Bickel for a given panel key."""
        if key == 'input':
            data = X[:n_used].astype(np.float32)
        elif key == 'output':
            data = layer_reps[-1]
        else:
            layer_i = int(key.split('_')[1]) - 1
            data = layer_reps[layer_i]
        if args.velocity:
            data = _with_velocity(data, n_used, seq_len)
        return data[lb_idx]

    # ---- Levina-Bickel on selected layers (optional) ----
    lb_results = {}
    if not args.no_lb:
        lb_ks = [10, 30, 100]
        lb_idx = rng.choice(n_used, min(args.lb_points, n_used), replace=False)
        print(f"\nLevina-Bickel on {len(lb_idx)} points:")

        for key in selected_keys:
            rep_lb = get_lb_data(key)
            lb_layer = {}
            for k in lb_ks:
                m, _ = levina_bickel_estimator(rep_lb, k)
                lb_layer[k] = m
            lb_results[key] = lb_layer
            lb_str = '  '.join(f'k={k}: {lb_layer[k]:.2f}' for k in lb_ks)
            label = key.replace('_', ' ').title()
            print(f"  {label} ({rep_lb.shape[1]}D):  {lb_str}")

    # ---- PCA pre-processing helper ----
    def _maybe_pca(data, title):
        """If --pca N is set and data has more than N dims, apply PCA."""
        if args.pca > 0 and data.shape[1] > args.pca:
            pca = PCA(n_components=args.pca)
            data_pca = pca.fit_transform(data)
            var = pca.explained_variance_ratio_.sum() * 100
            print(f"  PCA {data.shape[1]}D → {args.pca}D "
                  f"({var:.1f}% variance retained)")
            return data_pca, f'{title} → PCA {args.pca}D ({var:.0f}%)'
        return data, title

    # ---- UMAP for selected layers ----
    panels = [get_panel_data(key) for key in selected_keys]

    sil_results = {}

    # ---- Phase 1: GPU k-NN precomputation for all panels ----
    use_gpu_knn = torch.cuda.is_available()
    precomputed = {}
    if use_gpu_knn:
        n_panels = len(panels)
        print(f"\nComputing GPU k-NN for {n_panels} panels...")
        t_knn = time.perf_counter()
        for title, data, lb_key in panels:
            data_pca, _ = _maybe_pca(data, title)
            knn_dists, knn_indices = gpu_knn(
                data_pca, k=args.umap_neighbors, batch_size=2048)
            precomputed[lb_key] = (knn_indices, knn_dists)
        print(f"  All k-NN done in {time.perf_counter() - t_knn:.2f}s")

    # ---- Phase 2: UMAP layout ----
    if args.per_layer:
        for title, data, lb_key in panels:
            data, title = _maybe_pca(data, title)
            t_start = time.perf_counter()
            pre = precomputed.get(lb_key)
            print(f"Fitting UMAP for {title}...")
            embedding = _fit_umap(
                (title, data, lb_key, args.umap_neighbors, pre))
            print(f"  done in {time.perf_counter() - t_start:.2f}s")

            sil = gpu_silhouette_score(embedding, states_sub)
            sil_results[lb_key] = sil

            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=states_sub,
                            cmap='tab10', s=4, alpha=0.5)
            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)

            subtitle_parts = []
            if lb_key in lb_results:
                lb30 = lb_results[lb_key].get(30, None)
                if lb30 is not None:
                    subtitle_parts.append(f'LB k=30: {lb30:.1f}')
            subtitle_parts.append(f'Sil: {sil:.3f}')
            title += f'  ({", ".join(subtitle_parts)})'
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.2)
            cbar = plt.colorbar(sc, ax=ax, label='Circle index')
            cbar.set_ticks(range(config['n_circles']))

            plt.tight_layout()
            fname = f'umap_{lb_key}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {fname}")
    else:
        n_panels = len(panels)
        n_cols = min(n_panels, 3)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 6 * n_rows))
        if n_panels == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for ax_i, (title, data, lb_key) in enumerate(panels):
            data, title = _maybe_pca(data, title)
            t_start = time.perf_counter()
            pre = precomputed.get(lb_key)
            print(f"Fitting UMAP for {title}...")
            embedding = _fit_umap(
                (title, data, lb_key, args.umap_neighbors, pre))
            print(f"  done in {time.perf_counter() - t_start:.2f}s")

            sil = gpu_silhouette_score(embedding, states_sub)
            sil_results[lb_key] = sil

            ax = axes[ax_i]
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=states_sub,
                            cmap='tab10', s=3, alpha=0.5)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')

            subtitle_parts = []
            if lb_key in lb_results:
                lb30 = lb_results[lb_key].get(30, None)
                if lb30 is not None:
                    subtitle_parts.append(f'LB k=30: {lb30:.1f}')
            subtitle_parts.append(f'Sil: {sil:.3f}')
            title += f'  ({", ".join(subtitle_parts)})'
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.2)

        cbar = plt.colorbar(sc, ax=axes[len(panels) - 1],
                             label='Circle index')
        cbar.set_ticks(range(config['n_circles']))

        for ax_i in range(len(panels), len(axes)):
            axes[ax_i].axis('off')

        fig.suptitle('UMAP of Intermediate Representations', fontsize=15,
                     fontweight='bold', y=1.02)
        plt.tight_layout()

        fname = ('representation_umap_velocity.png' if args.velocity
                 else 'representation_umap.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved {fname}")

    # ---- Print silhouette summary ----
    if sil_results:
        print(f"\nSilhouette scores (UMAP 2D):")
        for key in selected_keys:
            if key in sil_results:
                label = key.replace('_', ' ').title()
                print(f"  {label}: {sil_results[key]:.3f}")

    # Hidden-state UMAP disabled — was computing UMAP of Mamba SSM outputs
    # before residual add. Re-enable by uncommenting the block below.
    # raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    # if hasattr(raw_model, 'layers'):
    #     ... (see git history)


if __name__ == '__main__':
    main()
