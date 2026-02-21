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
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
from sklearn.decomposition import PCA

from masked_model_gpu_mamba_next import CausalTimeSeriesMamba
from estimate_dimension import levina_bickel_estimator


def load_model(checkpoint_path, device):
    """Load a trained CausalTimeSeriesMamba from checkpoint."""
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

    if not hasattr(raw_model, 'layers'):
        raise ValueError("Model does not have .layers — hidden state "
                         "extraction requires CausalTimeSeriesMamba")

    n_steps = X.shape[0]
    n_windows = n_steps // seq_len
    n_used = n_windows * seq_len
    X_trim = X[:n_used]

    n_layers = len(raw_model.layers)
    all_hidden = [[] for _ in range(n_layers)]

    # Storage for hook captures (reset each batch)
    captured = [None] * n_layers

    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output.detach()
        return hook

    # Register hooks on the mamba sub-module inside each MambaLayer
    hooks = []
    for i, layer in enumerate(raw_model.layers):
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
    # Last entry is the output projection (feature_dim D), rest are encoder
    # layers (d_model D)
    n_encoder_layers = len(layer_reps) - 1
    n_all = len(layer_reps)

    print(f"Extracted {n_encoder_layers} encoder layers + output projection")
    print(f"  Encoder layers: {layer_reps[0].shape}")
    print(f"  Output projection: {layer_reps[-1].shape}")

    # ---- Parse --layers filter ----
    # Build the full list of available panels: input, layer_1..N, output
    all_panel_keys = ['input']
    for i in range(n_encoder_layers):
        all_panel_keys.append(f'layer_{i+1}')
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

    # ---- Build panel data for selected layers ----
    def get_panel_data(key):
        """Return (title, data_subsampled, key) for a given panel key."""
        if key == 'input':
            return (f'Input ({X.shape[1]}D)', X[idx].astype(np.float32), key)
        elif key == 'output':
            return (f'Output ({layer_reps[-1].shape[1]}D)',
                    layer_reps[-1][idx], key)
        else:
            # layer_N → index N-1
            layer_i = int(key.split('_')[1]) - 1
            return (f'Layer {layer_i+1} ({layer_reps[layer_i].shape[1]}D)',
                    layer_reps[layer_i][idx], key)

    def get_lb_data(key):
        """Return data for Levina-Bickel for a given panel key."""
        if key == 'input':
            return X[lb_idx].astype(np.float32)
        elif key == 'output':
            return layer_reps[-1][lb_idx]
        else:
            layer_i = int(key.split('_')[1]) - 1
            return layer_reps[layer_i][lb_idx]

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

    def _make_single_umap(title, data, lb_key, save_path):
        """Compute UMAP and save a single-layer figure."""
        data, title = _maybe_pca(data, title)
        print(f"Computing UMAP for {title}...")
        reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=0.3,
                            metric='euclidean', n_jobs=-1)
        embedding = reducer.fit_transform(data)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=states_sub,
                        cmap='tab10', s=4, alpha=0.5)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)

        if lb_key in lb_results:
            lb30 = lb_results[lb_key].get(30, None)
            if lb30 is not None:
                title += f'  (LB k=30: {lb30:.1f})'
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.2)
        cbar = plt.colorbar(sc, ax=ax, label='Circle index')
        cbar.set_ticks(range(config['n_circles']))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")
        return embedding

    if args.per_layer:
        for title, data, lb_key in panels:
            fname = f'umap_{lb_key}.png'
            _make_single_umap(title, data, lb_key, fname)
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
            print(f"Computing UMAP for {title}...")
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=0.3,
                                metric='euclidean', n_jobs=-1)
            embedding = reducer.fit_transform(data)

            ax = axes[ax_i]
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=states_sub,
                            cmap='tab10', s=3, alpha=0.5)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')

            if lb_key in lb_results:
                lb30 = lb_results[lb_key].get(30, None)
                if lb30 is not None:
                    title += f'  (LB k=30: {lb30:.1f})'
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

        fname = 'representation_umap.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved {fname}")

    # ---- Hidden-state UMAP (Mamba SSM outputs before residual) ----
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(raw_model, 'layers'):
        print("\n--- Extracting Mamba hidden states (before residual) ---")
        hidden_reps, _ = extract_hidden_states(
            model, X, device, batch_size=64, seq_len=seq_len,
        )
        n_hidden = len(hidden_reps)
        n_cols_h = min(n_hidden, 4)
        n_rows_h = (n_hidden + n_cols_h - 1) // n_cols_h

        fig, axes = plt.subplots(n_rows_h, n_cols_h,
                                 figsize=(7 * n_cols_h, 6 * n_rows_h))
        if n_hidden == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for i in range(n_hidden):
            data_h = hidden_reps[i][idx]
            title_h = f'Hidden {i+1} ({data_h.shape[1]}D)'
            data_h, title_h = _maybe_pca(data_h, title_h)
            print(f"Computing UMAP for {title_h}...")
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=0.3,
                                metric='euclidean', n_jobs=-1)
            emb = reducer.fit_transform(data_h)

            ax = axes[i]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=states_sub,
                            cmap='tab10', s=3, alpha=0.5)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(title_h, fontsize=12)
            ax.grid(True, alpha=0.2)

        cbar = plt.colorbar(sc, ax=axes[n_hidden - 1], label='Circle index')
        cbar.set_ticks(range(config['n_circles']))

        for ax_i in range(n_hidden, len(axes)):
            axes[ax_i].axis('off')

        fig.suptitle('UMAP of Mamba Hidden States (before residual)',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        fname_h = 'representation_umap_hidden.png'
        plt.savefig(fname_h, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved {fname_h}")


if __name__ == '__main__':
    main()
