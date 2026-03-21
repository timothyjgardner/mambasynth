"""
Unsupervised clustering of learned representations.

Extracts intermediate representations from each Mamba layer, runs
clustering (spectral, k-means, GMM, or Ward), and visualises the
discovered clusters on UMAP.

Usage
-----
    python cluster_representations.py --checkpoint model_search_lam1.0_temp0.3.pt --umap
    python cluster_representations.py --checkpoint model_search_lam1.0_temp0.3.pt --n-neighbors 50 --umap
    python cluster_representations.py --n-clusters 15 --layers 3,4,5 --umap
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from masked_model_gpu_mamba_next import CausalTimeSeriesMamba
from evaluate_representations import load_model, extract_representations


def with_velocity(rep, n_used, seq_len):
    """Concatenate finite-difference velocity with position.

    Zeros out velocity at window boundaries where the difference
    would span two non-contiguous windows.
    """
    vel = np.zeros_like(rep)
    vel[:-1] = rep[1:] - rep[:-1]
    for w in range(n_used // seq_len):
        vel[w * seq_len + seq_len - 1] = 0.0
    return np.concatenate([rep, vel], axis=1)


def preprocess(reps, normalize=False, pca_dim=0):
    """Optional L2 normalization and/or PCA before clustering."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize as l2_normalize
    if normalize:
        reps = l2_normalize(reps)
    if pca_dim > 0 and reps.shape[1] > pca_dim:
        reps = PCA(n_components=pca_dim, random_state=42).fit_transform(reps)
    return reps


def _gpu_knn(X_np, k, batch_size=2048):
    """Batched k-NN on GPU using PyTorch.

    Computes pairwise L2 distances in batches to avoid O(n²) memory.
    Returns (distances, indices) as numpy arrays.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.from_numpy(X_np).float().to(device)
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


def _gpu_knn_cross(X_query, X_ref, k, batch_size=2048):
    """Batched k-NN: for each query point, find k nearest in ref set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Q = torch.from_numpy(X_query).float().to(device)
    R = torch.from_numpy(X_ref).float().to(device)
    n = Q.shape[0]
    all_idx = torch.zeros(n, k, dtype=torch.long, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(Q[start:end], R)
        _, topk_i = dists.topk(k, dim=1, largest=False)
        all_idx[start:end] = topk_i

    return all_idx.cpu().numpy()


def _spectral_core(X, n_clusters, n_neighbors, seed):
    """Full spectral clustering on a (small) dataset. Returns labels."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from sklearn.preprocessing import normalize

    n = X.shape[0]
    dists, indices = _gpu_knn(X, n_neighbors)

    sigma = np.median(dists)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    W = csr_matrix((weights.ravel(), (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2

    d = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))),
                            shape=(n, n))
    L_norm = D_inv_sqrt @ W @ D_inv_sqrt

    _, eigenvectors = eigsh(L_norm, k=n_clusters, which='LM')
    embedding = normalize(eigenvectors, norm='l2', axis=1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(embedding)


class _GPUSpectral:
    """GPU-accelerated spectral clustering."""
    def __init__(self, n_clusters, n_neighbors, seed):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed

    def fit_predict(self, X):
        return _spectral_core(X, self.n_clusters, self.n_neighbors,
                              self.seed)


class _NystromSpectral:
    """Nyström-approximated spectral clustering.

    Runs full spectral on m landmark points, then extends eigenvectors
    to all n points via the Nyström formula.
    """
    def __init__(self, n_clusters, n_neighbors, seed, n_landmarks=5000):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.n_landmarks = n_landmarks

    def fit_predict(self, X):
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        from sklearn.preprocessing import normalize

        n = X.shape[0]
        m = min(self.n_landmarks, n)
        rng = np.random.RandomState(self.seed)

        idx_land = rng.choice(n, m, replace=False)
        X_land = X[idx_land]

        # Full spectral eigenvectors on landmarks
        dists_mm, indices_mm = _gpu_knn(X_land, self.n_neighbors)
        sigma = np.median(dists_mm)

        weights_mm = np.exp(-dists_mm ** 2 / (2 * sigma ** 2))
        rows = np.repeat(np.arange(m), self.n_neighbors)
        cols = indices_mm.ravel()
        W_mm = csr_matrix((weights_mm.ravel(), (rows, cols)), shape=(m, m))
        W_mm = (W_mm + W_mm.T) / 2

        d = np.array(W_mm.sum(axis=1)).flatten()
        d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv = csr_matrix((d_inv, (np.arange(m), np.arange(m))),
                           shape=(m, m))
        L_mm = D_inv @ W_mm @ D_inv

        eigenvalues, V_land = eigsh(L_mm, k=self.n_clusters, which='LM')

        # Nyström extension: compute affinity of all points to landmarks
        idx_cross = _gpu_knn_cross(X, X_land, self.n_neighbors)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.from_numpy(X).float().to(device)
        L_t = torch.from_numpy(X_land).float().to(device)
        dists_nm = torch.zeros(n, self.n_neighbors, device=device)
        for start in range(0, n, 2048):
            end = min(start + 2048, n)
            idx_batch = torch.from_numpy(idx_cross[start:end]).to(device)
            diff = X_t[start:end].unsqueeze(1) - L_t[idx_batch]
            dists_nm[start:end] = diff.pow(2).sum(-1).sqrt()
        dists_nm = dists_nm.cpu().numpy()

        weights_nm = np.exp(-dists_nm ** 2 / (2 * sigma ** 2))
        rows_nm = np.repeat(np.arange(n), self.n_neighbors)
        cols_nm = idx_cross.ravel()
        W_nm = csr_matrix((weights_nm.ravel(), (rows_nm, cols_nm)),
                          shape=(n, m))

        # Extend eigenvectors: V_all ≈ W_nm @ V_land @ diag(1/eigenvalues)
        V_ext = W_nm @ (V_land / eigenvalues[np.newaxis, :])
        embedding = normalize(V_ext, norm='l2', axis=1)
        km = KMeans(n_clusters=self.n_clusters, n_init=10,
                    random_state=self.seed)
        return km.fit_predict(embedding)


class _TwoStageSpectral:
    """Two-stage spectral clustering: spectral on subsample, then
    GPU k-NN propagation to assign all remaining points."""
    def __init__(self, n_clusters, n_neighbors, seed, n_subsample=10000):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.seed = seed
        self.n_subsample = n_subsample

    def fit_predict(self, X):
        n = X.shape[0]
        m = min(self.n_subsample, n)
        rng = np.random.RandomState(self.seed)

        idx_sub = rng.choice(n, m, replace=False)
        X_sub = X[idx_sub]

        labels_sub = _spectral_core(X_sub, self.n_clusters,
                                    self.n_neighbors, self.seed)

        # Propagate: assign each point to the cluster of its nearest
        # subsampled neighbor
        nn_idx = _gpu_knn_cross(X, X_sub, k=1)
        return labels_sub[nn_idx.ravel()]


def build_clusterer(method, n_clusters, n_neighbors, seed,
                    n_landmarks=5000, n_subsample=10000):
    """Return a clustering object with a .fit_predict() interface."""
    if method == 'kmeans':
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    elif method == 'spectral':
        return _GPUSpectral(n_clusters, n_neighbors, seed)
    elif method == 'nystrom':
        return _NystromSpectral(n_clusters, n_neighbors, seed,
                                n_landmarks=n_landmarks)
    elif method == 'twostage':
        return _TwoStageSpectral(n_clusters, n_neighbors, seed,
                                 n_subsample=n_subsample)
    elif method == 'spectral_cpu':
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=seed,
            n_jobs=-1,
        )
    elif method == 'gmm':
        return _GMMWrapper(n_clusters, seed)
    elif method == 'ward':
        return AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward',
        )
    else:
        raise ValueError(f"Unknown method: {method}")


class _GMMWrapper:
    """Thin wrapper so GaussianMixture has .fit_predict()."""
    def __init__(self, n_clusters, seed):
        self.gmm = GaussianMixture(
            n_components=n_clusters, covariance_type='full',
            n_init=3, random_state=seed,
        )

    def fit_predict(self, X):
        return self.gmm.fit_predict(X)


def cluster_layer(reps, n_clusters, method='spectral', n_neighbors=30,
                  seed=42, n_landmarks=5000, n_subsample=10000):
    """Cluster a single layer's representations."""
    clusterer = build_clusterer(method, n_clusters, n_neighbors, seed,
                                n_landmarks=n_landmarks,
                                n_subsample=n_subsample)
    pred = clusterer.fit_predict(reps)
    n_found = int(pred.max()) + 1
    sil = silhouette_score(reps, pred, sample_size=min(5000, len(reps)),
                           random_state=seed)
    return {'pred': pred, 'n_found': n_found, 'silhouette': sil}


def plot_summary(results, selected_keys, save_path):
    """Bar chart of silhouette and cluster count across layers."""
    labels = []
    for k in selected_keys:
        if k == 'input':
            labels.append('Input')
        elif k == 'output':
            labels.append('Output')
        else:
            labels.append(f'L{k.split("_")[1]}')

    sil = [results[k]['silhouette'] for k in selected_keys]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bars = ax.bar(x, sil, 0.5, color='#2196F3')

    for i, v in enumerate(sil):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
                fontsize=8, color='#2196F3')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Cluster Silhouette per Layer')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_umap_clusters(reps_dict, results, selected_keys, states,
                       n_classes, umap_neighbors, save_path,
                       umap_cache_path=None):
    """Two-row grid per layer block: top = clusters, bottom = ground truth.

    Same UMAP coordinates are used for both rows.
    If umap_cache_path is set, embeddings are saved/loaded from that file.
    """
    cached_embeddings = {}
    if umap_cache_path and Path(umap_cache_path).exists():
        loaded = np.load(umap_cache_path)
        cached_embeddings = {k: loaded[k] for k in loaded.files}
        print(f"  Loaded cached UMAP embeddings from {umap_cache_path}")

    n_panels = len(selected_keys)
    n_cols = min(n_panels, 3)
    n_rows_per_block = (n_panels + n_cols - 1) // n_cols
    total_rows = n_rows_per_block * 2

    fig, axes = plt.subplots(total_rows, n_cols,
                             figsize=(7 * n_cols, 5.5 * total_rows))
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    axes = np.atleast_2d(axes)

    def get_cmap(k):
        if k <= 10:
            return 'tab10'
        elif k <= 20:
            return 'tab20'
        else:
            return 'gist_ncar'

    new_embeddings = {}
    for panel_i, key in enumerate(selected_keys):
        row_in_block = panel_i // n_cols
        col = panel_i % n_cols

        reps = reps_dict[key]
        pred = results[key]['pred']
        n_found = results[key]['n_found']
        sil = results[key]['silhouette']

        if key == 'input':
            label = f'Input ({reps.shape[1]}D)'
        elif key == 'output':
            label = f'Output ({reps.shape[1]}D)'
        else:
            label = f'Layer {key.split("_")[1]} ({reps.shape[1]}D)'

        if key in cached_embeddings and cached_embeddings[key].shape[0] == reps.shape[0]:
            print(f"  Using cached UMAP for {label}")
            emb = cached_embeddings[key]
        else:
            print(f"  Computing UMAP for {label}...")
            reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3,
                                metric='euclidean', n_jobs=-1)
            emb = reducer.fit_transform(reps)
        new_embeddings[key] = emb

        # Top block: clusters
        ax_clust = axes[row_in_block, col]
        cmap = get_cmap(n_found)
        sc_c = ax_clust.scatter(emb[:, 0], emb[:, 1], c=pred, cmap=cmap,
                                s=3, alpha=0.5, vmin=0,
                                vmax=max(n_found - 1, 1))
        ax_clust.set_title(f'{label}  (k={n_found}, sil={sil:.3f})',
                           fontsize=11)
        ax_clust.set_xlabel('UMAP 1', fontsize=8)
        ax_clust.set_ylabel('UMAP 2', fontsize=8)
        ax_clust.grid(True, alpha=0.15)

        # Bottom block: ground truth
        ax_true = axes[n_rows_per_block + row_in_block, col]
        sc_t = ax_true.scatter(emb[:, 0], emb[:, 1], c=states, cmap='tab10',
                               s=3, alpha=0.5, vmin=0,
                               vmax=n_classes - 1)
        ax_true.set_title(f'{label} — Ground Truth', fontsize=11)
        ax_true.set_xlabel('UMAP 1', fontsize=8)
        ax_true.set_ylabel('UMAP 2', fontsize=8)
        ax_true.grid(True, alpha=0.15)

    # Hide unused axes
    for panel_i in range(n_panels, n_rows_per_block * n_cols):
        row_in_block = panel_i // n_cols
        col = panel_i % n_cols
        axes[row_in_block, col].axis('off')
        axes[n_rows_per_block + row_in_block, col].axis('off')

    # Colorbars on last used column
    last_col = (n_panels - 1) % n_cols
    last_row = (n_panels - 1) // n_cols
    plt.colorbar(sc_c, ax=axes[last_row, last_col],
                 label='Cluster', shrink=0.8)
    plt.colorbar(sc_t, ax=axes[n_rows_per_block + last_row, last_col],
                 label='Circle', shrink=0.8)

    fig.text(0.01, 0.75, 'Clusters', fontsize=14, fontweight='bold',
             rotation=90, va='center', ha='left')
    fig.text(0.01, 0.25, 'Ground Truth', fontsize=14, fontweight='bold',
             rotation=90, va='center', ha='left')

    fig.suptitle('UMAP: Unsupervised Clusters vs. Ground Truth',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

    if umap_cache_path:
        np.savez(umap_cache_path, **new_embeddings)
        print(f"  Saved UMAP embeddings to {umap_cache_path}")


def build_transition_matrix(labels, seq_len):
    """Build a transition count matrix, skipping window boundaries."""
    k = int(labels.max()) + 1
    T = np.zeros((k, k), dtype=np.float64)
    n = len(labels)
    for t in range(n - 1):
        if (t + 1) % seq_len == 0:
            continue
        T[labels[t], labels[t + 1]] += 1
    return T


def transition_to_probability(T):
    """Row-normalize transition counts to probabilities."""
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def merge_clusters_by_transitions(T, n_merge):
    """Merge clusters using hierarchical clustering on the transition matrix.

    Clusters that frequently transition to each other (same circle,
    different phases) get merged first.

    Returns mapping array: merged[old_label] = new_label
    """
    k = T.shape[0]
    P = transition_to_probability(T)
    # Symmetrize: average of P[i,j] and P[j,i] as similarity
    S = (P + P.T) / 2
    np.fill_diagonal(S, 0)
    # Convert to distance
    D = 1.0 - S
    np.fill_diagonal(D, 0)
    D = np.clip(D, 0, None)

    # Condensed distance matrix for scipy
    from scipy.spatial.distance import squareform
    D_condensed = squareform(D, checks=False)

    Z = linkage(D_condensed, method='average')
    group_labels = fcluster(Z, t=n_merge, criterion='maxclust')

    # Remap to 0-based contiguous labels
    unique = np.unique(group_labels)
    remap = {old: new for new, old in enumerate(unique)}
    merged = np.array([remap[g] for g in group_labels])
    return merged


def plot_transition_matrices(T_before, T_after, merge_map, save_path):
    """Plot before and after transition probability matrices side by side."""
    P_before = transition_to_probability(T_before)
    P_after = transition_to_probability(T_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    k_before = P_before.shape[0]
    im1 = ax1.imshow(P_before, cmap='Blues', vmin=0,
                     vmax=max(P_before.max(), 0.01))
    ax1.set_title(f'Before merging (k={k_before})', fontsize=12)
    ax1.set_xlabel('To cluster')
    ax1.set_ylabel('From cluster')
    ax1.set_xticks(range(k_before))
    ax1.set_yticks(range(k_before))
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='P(transition)')

    k_after = P_after.shape[0]
    im2 = ax2.imshow(P_after, cmap='Blues', vmin=0,
                     vmax=max(P_after.max(), 0.01))
    ax2.set_title(f'After merging (k={k_after})', fontsize=12)
    ax2.set_xlabel('To cluster')
    ax2.set_ylabel('From cluster')
    ax2.set_xticks(range(k_after))
    ax2.set_yticks(range(k_after))
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='P(transition)')

    # Annotate cells
    for ax, P in [(ax1, P_before), (ax2, P_after)]:
        n = P.shape[0]
        for i in range(n):
            for j in range(n):
                v = P[i, j]
                if v > 0.005:
                    color = 'white' if v > P.max() * 0.5 else 'black'
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=7 if n > 12 else 8, color=color)

    fig.suptitle('Cluster Transition Probabilities', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def apply_merge(labels, merge_map):
    """Remap cluster labels according to merge_map."""
    return merge_map[labels]


def plot_signal_panels(X_full, states_full, layer_reps_full, reps_sub, pred_sub,
                       idx_sub, n_used, seq_len, layer_key, n_panels,
                       n_found, save_path, seed=42):
    """Show raw input signals with cluster and ground-truth color bars.

    Uses k-NN from the clustered subsample to label every timestep in
    a few contiguous windows.
    """
    # Build k-NN classifier from subsampled reps → cluster labels
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(reps_sub, pred_sub)

    # Get full (un-subsampled) reps for the chosen layer
    if layer_key == 'input':
        full_reps = X_full[:n_used].astype(np.float32)
    elif layer_key == 'output':
        full_reps = layer_reps_full[-1]
    else:
        li = int(layer_key.split('_')[1]) - 1
        full_reps = layer_reps_full[li]

    # Pick random windows
    n_windows = n_used // seq_len
    rng = np.random.default_rng(seed)
    win_indices = rng.choice(n_windows, size=min(n_panels, n_windows),
                             replace=False)
    win_indices.sort()

    cmap_cluster = plt.cm.tab10 if n_found <= 10 else plt.cm.tab20

    for panel_i, wi in enumerate(win_indices):
        t0 = wi * seq_len
        t1 = t0 + seq_len
        x_win = X_full[t0:t1]
        states_win = states_full[t0:t1]
        reps_win = full_reps[t0:t1]

        cluster_win = knn.predict(reps_win)

        vmax = max(abs(x_win.min()), abs(x_win.max()))

        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 6],
                              width_ratios=[1, 0.02], hspace=0.08,
                              wspace=0.03)

        # Ground truth bar
        ax_gt = fig.add_subplot(gs[0, 0])
        for t in range(seq_len):
            ax_gt.axvspan(t, t + 1,
                          color=plt.cm.tab10(states_win[t] / 10),
                          alpha=0.8, linewidth=0)
        ax_gt.set_xlim(0, seq_len)
        ax_gt.set_yticks([])
        ax_gt.set_ylabel('True', fontsize=9)
        ax_gt.set_title(f'Window {wi} (t={t0}–{t1})  —  '
                        f'{layer_key.replace("_", " ").title()}',
                        fontsize=12)
        plt.setp(ax_gt.get_xticklabels(), visible=False)

        # Cluster bar
        ax_cl = fig.add_subplot(gs[1, 0], sharex=ax_gt)
        for t in range(seq_len):
            c = cluster_win[t]
            ax_cl.axvspan(t, t + 1,
                          color=cmap_cluster(c / max(n_found - 1, 1)),
                          alpha=0.8, linewidth=0)
        ax_cl.set_xlim(0, seq_len)
        ax_cl.set_yticks([])
        ax_cl.set_ylabel('Cluster', fontsize=9)
        plt.setp(ax_cl.get_xticklabels(), visible=False)

        # Signal heatmap
        ax_sig = fig.add_subplot(gs[2, 0], sharex=ax_gt)
        im = ax_sig.imshow(
            x_win.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, x_win.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_sig.set_xlabel('Time step', fontsize=9)
        ax_sig.set_ylabel('Dimension', fontsize=9)

        ax_cb = fig.add_subplot(gs[2, 1])
        plt.colorbar(im, cax=ax_cb)
        fig.add_subplot(gs[0, 1]).axis('off')
        fig.add_subplot(gs[1, 1]).axis('off')

        fname = save_path.replace('.png', f'_{panel_i}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


def plot_signal_panels_direct(X_full, states_full, all_labels, n_used,
                              seq_len, layer_key, n_panels, n_found,
                              save_path, seed=42):
    """Signal panels using pre-computed labels for every timestep."""
    n_windows = n_used // seq_len
    rng = np.random.default_rng(seed)
    win_indices = rng.choice(n_windows, size=min(n_panels, n_windows),
                             replace=False)
    win_indices.sort()

    cmap_cluster = plt.cm.tab10 if n_found <= 10 else plt.cm.tab20

    for panel_i, wi in enumerate(win_indices):
        t0 = wi * seq_len
        t1 = t0 + seq_len
        x_win = X_full[t0:t1]
        states_win = states_full[t0:t1]
        cluster_win = all_labels[t0:t1]

        vmax = max(abs(x_win.min()), abs(x_win.max()))

        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 6],
                              width_ratios=[1, 0.02], hspace=0.08,
                              wspace=0.03)

        ax_gt = fig.add_subplot(gs[0, 0])
        for t in range(seq_len):
            ax_gt.axvspan(t, t + 1,
                          color=plt.cm.tab10(states_win[t] / 10),
                          alpha=0.8, linewidth=0)
        ax_gt.set_xlim(0, seq_len)
        ax_gt.set_yticks([])
        ax_gt.set_ylabel('True', fontsize=9)
        ax_gt.set_title(f'Window {wi} (t={t0}–{t1})  —  '
                        f'{layer_key.replace("_", " ").title()} (merged)',
                        fontsize=12)
        plt.setp(ax_gt.get_xticklabels(), visible=False)

        ax_cl = fig.add_subplot(gs[1, 0], sharex=ax_gt)
        for t in range(seq_len):
            c = cluster_win[t]
            ax_cl.axvspan(t, t + 1,
                          color=cmap_cluster(c / max(n_found - 1, 1)),
                          alpha=0.8, linewidth=0)
        ax_cl.set_xlim(0, seq_len)
        ax_cl.set_yticks([])
        ax_cl.set_ylabel('Cluster', fontsize=9)
        plt.setp(ax_cl.get_xticklabels(), visible=False)

        ax_sig = fig.add_subplot(gs[2, 0], sharex=ax_gt)
        im = ax_sig.imshow(
            x_win.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, x_win.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_sig.set_xlabel('Time step', fontsize=9)
        ax_sig.set_ylabel('Dimension', fontsize=9)

        ax_cb = fig.add_subplot(gs[2, 1])
        plt.colorbar(im, cax=ax_cb)
        fig.add_subplot(gs[0, 1]).axis('off')
        fig.add_subplot(gs[1, 1]).axis('off')

        fname = save_path.replace('.png', f'_{panel_i}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description='Unsupervised clustering of Mamba representations.')
    parser.add_argument('--checkpoint', type=str, default='bert_model.pt')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-clusters', type=int, default=None,
                        help='Number of clusters (default: n_circles from '
                             'data config)')
    parser.add_argument('--points', type=int, default=10000,
                        help='Max points to subsample')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layers (e.g. "4,5" or '
                             '"input,4,5,output"). Default: all.')
    parser.add_argument('--method', type=str, default='spectral',
                        choices=['kmeans', 'spectral', 'nystrom',
                                 'twostage', 'spectral_cpu', 'gmm', 'ward'],
                        help='Clustering algorithm. spectral=full GPU, '
                             'nystrom=Nystrom approx, twostage=subsample+'
                             'propagate, spectral_cpu=sklearn CPU fallback.')
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help='Neighbors for spectral clustering affinity '
                             'graph (default: 30)')
    parser.add_argument('--n-landmarks', type=int, default=5000,
                        help='Number of landmark points for Nystrom '
                             'approximation (default: 5000)')
    parser.add_argument('--n-subsample', type=int, default=10000,
                        help='Number of subsample points for two-stage '
                             'spectral clustering (default: 10000)')
    parser.add_argument('--velocity', action='store_true',
                        help='Concatenate finite-difference velocity with '
                             'position, doubling the dimensionality')
    parser.add_argument('--normalize', action='store_true',
                        help='L2-normalize representations before clustering')
    parser.add_argument('--pca', type=int, default=0, metavar='N',
                        help='PCA reduce to N dimensions before clustering '
                             '(0 = no PCA)')
    parser.add_argument('--umap', action='store_true',
                        help='Produce UMAP cluster visualisation')
    parser.add_argument('--umap-neighbors', type=int, default=50,
                        help='UMAP n_neighbors parameter (default: 50)')
    parser.add_argument('--umap-cache', type=str, default=None,
                        metavar='PATH',
                        help='Path to .npz file for saving/loading UMAP '
                             'embeddings. If the file exists, embeddings are '
                             'loaded instead of recomputed.')
    parser.add_argument('--merge', type=int, default=0, metavar='N',
                        help='Merge clusters down to N groups using '
                             'transition matrix (0 = no merging)')
    parser.add_argument('--signal-panels', type=int, default=0, metavar='N',
                        help='Show N windows of the raw input signal with '
                             'cluster color bars')
    parser.add_argument('--signal-layer', type=str, default=None,
                        help='Layer to use for signal panel clustering '
                             '(e.g. "4"). Default: best silhouette layer.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    npz = np.load(data_dir / 'data.npz')
    X = npz['X']
    states = npz['states']
    with open(data_dir / 'config.json') as f:
        config = json.load(f)

    n_classes = config['n_circles']
    n_clusters = args.n_clusters or n_classes
    print(f"Data: {X.shape[0]} steps, {n_classes} circles, "
          f"clustering with k={n_clusters}, method={args.method}")

    model, model_args = load_model(args.checkpoint, device)
    seq_len = model_args['seq_len']

    layer_reps, n_used = extract_representations(
        model, X, device, batch_size=64, seq_len=seq_len,
    )
    states_used = states[:n_used]
    n_encoder_layers = len(layer_reps) - 1

    # Build panel keys
    all_keys = ['input']
    for i in range(n_encoder_layers):
        all_keys.append(f'layer_{i + 1}')
    all_keys.append('output')

    if args.layers is not None:
        selected_keys = []
        for token in args.layers.split(','):
            token = token.strip().lower()
            if token in ('input', 'output'):
                selected_keys.append(token)
            else:
                key = f'layer_{int(token)}'
                if key in all_keys:
                    selected_keys.append(key)
                else:
                    print(f"WARNING: layer {token} not found, skipping")
        if not selected_keys:
            selected_keys = all_keys
    else:
        selected_keys = all_keys

    # Subsample
    rng = np.random.default_rng(args.seed)
    if n_used > args.points:
        idx = rng.choice(n_used, args.points, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_used)

    def get_reps(key):
        if key == 'input':
            full = X[:n_used].astype(np.float32)
        elif key == 'output':
            full = layer_reps[-1]
        else:
            li = int(key.split('_')[1]) - 1
            full = layer_reps[li]
        if args.velocity:
            full = with_velocity(full, n_used, seq_len)
        raw = full[idx]
        return preprocess(raw, normalize=args.normalize, pca_dim=args.pca)

    prep_parts = []
    if args.velocity:
        prep_parts.append('velocity')
    if args.normalize:
        prep_parts.append('L2-norm')
    if args.pca > 0:
        prep_parts.append(f'PCA→{args.pca}D')
    if prep_parts:
        print(f"Preprocessing: {' + '.join(prep_parts)}")

    # Cluster each layer
    results = {}
    reps_dict = {}
    header = f"{'Layer':<10} {'k':>4} {'Silhouette':>11}"
    print(f"\n{header}")
    print('-' * len(header))

    for key in selected_keys:
        reps = get_reps(key)
        reps_dict[key] = reps
        res = cluster_layer(reps, n_clusters, method=args.method,
                            n_neighbors=args.n_neighbors, seed=args.seed,
                            n_landmarks=args.n_landmarks,
                            n_subsample=args.n_subsample)
        results[key] = res

        if key == 'input':
            label = 'Input'
        elif key == 'output':
            label = 'Output'
        else:
            label = f'Layer {key.split("_")[1]}'

        print(f"{label:<10} {res['n_found']:>4} {res['silhouette']:>11.3f}")

    # ---- Transition-based merging ----
    if args.merge > 0:
        # Pick which layer to merge on
        if args.signal_layer is not None:
            token = args.signal_layer.strip().lower()
            merge_key = token if token in ('input', 'output') \
                else f'layer_{int(token)}'
        else:
            merge_key = max(results, key=lambda k: results[k]['silhouette'])

        merge_label = merge_key.replace('_', ' ').title()
        print(f"\nMerging clusters on {merge_label} via transition matrix...")

        # Label ALL timesteps via k-NN from subsample
        knn_merge = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn_merge.fit(reps_dict[merge_key], results[merge_key]['pred'])

        if merge_key == 'input':
            full_reps_merge = X[:n_used].astype(np.float32)
        elif merge_key == 'output':
            full_reps_merge = layer_reps[-1]
        else:
            li = int(merge_key.split('_')[1]) - 1
            full_reps_merge = layer_reps[li]

        all_labels = knn_merge.predict(full_reps_merge)

        # Build transition matrix on full temporal sequence
        k_before = results[merge_key]['n_found']
        T_before = build_transition_matrix(all_labels, seq_len)

        # Merge
        merge_map = merge_clusters_by_transitions(T_before, args.merge)
        print(f"  Merge map: {dict(enumerate(merge_map))}")

        # Apply merge to all labels and to the subsample
        all_labels_merged = apply_merge(all_labels, merge_map)
        sub_labels_merged = apply_merge(
            results[merge_key]['pred'], merge_map)

        # Build merged transition matrix
        T_after = build_transition_matrix(all_labels_merged, seq_len)
        n_after = int(all_labels_merged.max()) + 1

        # Update results with merged labels
        results[merge_key]['pred_before_merge'] = results[merge_key]['pred']
        results[merge_key]['pred'] = sub_labels_merged
        results[merge_key]['n_found'] = n_after
        results[merge_key]['merge_map'] = merge_map
        results[merge_key]['all_labels_merged'] = all_labels_merged

    # Output suffix
    ckpt_stem = Path(args.checkpoint).stem
    suffix = ckpt_stem.replace('model_search_', '').replace('model_', '')
    suffix += f'_{args.method}'
    if args.method == 'spectral':
        suffix += f'_nn{args.n_neighbors}'
    if args.velocity:
        suffix += '_vel'
    if args.normalize:
        suffix += '_norm'
    if args.pca > 0:
        suffix += f'_pca{args.pca}'
    suffix += f'_k{n_clusters}'
    if args.merge > 0:
        suffix += f'_merge{args.merge}'

    plot_summary(results, selected_keys,
                 f'cluster_summary_{suffix}.png')

    # Plot transition matrices if merging was done
    if args.merge > 0:
        plot_transition_matrices(
            T_before, T_after, merge_map,
            f'cluster_transitions_{suffix}.png')

    if args.umap:
        print(f"\nComputing UMAP cluster plots...")
        states_sub = states_used[idx]
        umap_cache = args.umap_cache
        if umap_cache is None:
            umap_cache = f'umap_cache_{suffix}.npz'
        plot_umap_clusters(reps_dict, results, selected_keys, states_sub,
                           n_classes, args.umap_neighbors,
                           f'cluster_umap_{suffix}.png',
                           umap_cache_path=umap_cache)

    if args.signal_panels > 0:
        # Pick which layer to use
        if args.signal_layer is not None:
            token = args.signal_layer.strip().lower()
            if token in ('input', 'output'):
                sig_key = token
            else:
                sig_key = f'layer_{int(token)}'
        else:
            sig_key = max(results, key=lambda k: results[k]['silhouette'])

        if sig_key not in results:
            print(f"WARNING: {sig_key} not in evaluated layers, "
                  f"using {list(results.keys())[0]}")
            sig_key = list(results.keys())[0]

        sig_label = sig_key.replace('_', ' ').title()
        print(f"\nGenerating signal panels using {sig_label}...")

        # If merging was done and this is the merge layer, use pre-computed
        # all_labels_merged directly for signal panels
        if args.merge > 0 and 'all_labels_merged' in results.get(sig_key, {}):
            plot_signal_panels_direct(
                X, states, results[sig_key]['all_labels_merged'],
                n_used, seq_len, sig_key, args.signal_panels,
                results[sig_key]['n_found'],
                f'cluster_signal_{suffix}.png', seed=args.seed,
            )
        else:
            plot_signal_panels(
                X, states, layer_reps, reps_dict[sig_key],
                results[sig_key]['pred'], idx, n_used, seq_len,
                sig_key, args.signal_panels, results[sig_key]['n_found'],
                f'cluster_signal_{suffix}.png', seed=args.seed,
            )


if __name__ == '__main__':
    main()
