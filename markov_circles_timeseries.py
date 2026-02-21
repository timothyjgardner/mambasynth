"""
Markov-Switching Time Series on Multiple Circles

Generates a time series where a point traverses 10 circles embedded in
20-dimensional ambient space.  All circles share the same center (the
origin) and have statistically similar radii (large enough that the
circular signal spans the full ambient space).  Each circle lives in
its own random 2D sub-plane and has a distinct, fixed angular velocity.

Dynamics
--------
The model visits one circle at a time as a "syllable":
  1. Enter circle i at its fixed entry angle.
  2. Traverse at the circle's fixed angular velocity  ω_i = 2π / period_i.
     Periods are linearly spaced from 40 steps (fastest) to 400 steps
     (slowest) per revolution — a 10× speed range.
  3. The number of complete revolutions K per visit is drawn so that
     K × period_i ≈ 400 steps (target from Normal(400, 100), quantised
     to whole revolutions).  This keeps dwell times near ~400 steps,
     preserves fixed entry/exit angles, and guarantees constant ω
     within each state.
  4. On exit, choose the next circle from the off-diagonal entries of a
     sparse transition matrix (ring + shortcut connectivity, zero
     diagonal).

Overlap control  (--subspace-dim)
---------------------------------
By default each circle's 2D plane is drawn from the full 20D ambient
space, so planes are nearly orthogonal and trajectories don't overlap.
The ``--subspace-dim`` flag restricts all planes to a shared subspace
of the given dimension, forcing geometric overlap:

    --subspace-dim 20   (default) minimal overlap — planes span full 20D
    --subspace-dim 4    significant overlap — 10 circles in 4D
    --subspace-dim 2    maximum overlap — all circles coplanar
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import umap


# ---------------------------------------------------------------------------
# Circle geometry
# ---------------------------------------------------------------------------

def create_circles(n_circles=10, ambient_dim=20, radius_mean=3.0,
                    radius_std=0.3, subspace_dim=None, seed=42):
    """
    Define n_circles circles in high-dimensional ambient space.

    All circles share the same center (the origin).  Each lives in its
    own random 2D sub-plane.  Radii are drawn from a narrow Normal
    distribution so they are statistically similar.

    Note: callers should set ``radius_mean`` large enough that the
    circular signal spans the ambient dimensions (e.g. 20.0 for a
    20-dimensional space, giving per-dimension amplitude ~20/√20 ≈ 4.5).

    Parameters
    ----------
    n_circles : int
        Number of circles.
    ambient_dim : int
        Dimension of the ambient space.
    radius_mean : float
        Mean radius for all circles.
    radius_std : float
        Std-dev of the radius distribution.
    subspace_dim : int or None
        Dimension of the subspace from which circle planes are drawn.
        Controls geometric overlap between circles:

        - ``None`` or ``ambient_dim`` (default): each circle's 2D plane
          is drawn from the full ambient space.  In high dimensions the
          planes are nearly orthogonal and trajectories barely overlap.
        - ``2``: all circles share the *same* 2D plane — maximum overlap
          (concentric rings with noise bleeding across).
        - ``3–6``: circles are forced into a small subspace and their
          planes significantly overlap, producing actual trajectory
          crossings once noise is added.

        Range: 2 ≤ subspace_dim ≤ ambient_dim.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    radii : ndarray (n_circles,)
    planes : list of (v1, v2) orthonormal basis pairs
    centers : list of ndarray (ambient_dim,)  – all identical (origin)
    """
    rng = np.random.default_rng(seed)

    if subspace_dim is None:
        subspace_dim = ambient_dim
    subspace_dim = max(2, min(subspace_dim, ambient_dim))

    # Similar radii drawn from a tight distribution
    radii = rng.normal(radius_mean, radius_std, n_circles)
    radii = np.clip(radii, radius_mean * 0.5, radius_mean * 1.5)  # safety

    # If subspace_dim < ambient_dim, generate a random orthonormal basis
    # that spans a subspace_dim-dimensional subspace of R^ambient_dim.
    # All circle planes will be drawn from within this subspace, forcing
    # geometric overlap.
    if subspace_dim < ambient_dim:
        M = rng.standard_normal((ambient_dim, subspace_dim))
        Q, _ = np.linalg.qr(M)
        subspace_basis = Q  # shape (ambient_dim, subspace_dim)
    else:
        subspace_basis = None  # full ambient space — no projection needed

    planes = []
    for _ in range(n_circles):
        if subspace_basis is not None:
            # Draw a 2D plane within the restricted subspace
            c1 = rng.standard_normal(subspace_dim)
            v1 = subspace_basis @ c1
            v1 /= np.linalg.norm(v1)
            c2 = rng.standard_normal(subspace_dim)
            v2 = subspace_basis @ c2
            v2 -= v2.dot(v1) * v1
            v2 /= np.linalg.norm(v2)
        else:
            # Full ambient space (original behaviour)
            v1 = rng.standard_normal(ambient_dim)
            v1 /= np.linalg.norm(v1)
            v2 = rng.standard_normal(ambient_dim)
            v2 -= v2.dot(v1) * v1
            v2 /= np.linalg.norm(v2)
        planes.append((v1, v2))

    # Shared center at the origin for all circles
    center = np.zeros(ambient_dim)
    centers = [center] * n_circles

    return radii, planes, centers


def point_on_circle(theta, radius, plane, center):
    """Map an angle to a point on a circle in ambient space."""
    v1, v2 = plane
    return center + radius * (np.cos(theta) * v1 + np.sin(theta) * v2)


# ---------------------------------------------------------------------------
# Sparse Markov transition matrix
# ---------------------------------------------------------------------------

def create_sparse_transition_matrix(n_circles=10, seed=42):
    """
    Build a sparse, row-stochastic *off-diagonal* transition matrix.

    The diagonal is zero because self-transitions are not used — the model
    always completes an integer number of full revolutions on a circle
    before switching to a different one.

    Structure: ring connectivity + long-range shortcut edges.

    Returns
    -------
    T : ndarray (n_circles, n_circles)
        Row-stochastic with zeros on the diagonal.
    """
    rng = np.random.default_rng(seed)
    T = np.zeros((n_circles, n_circles))

    # Ring neighbors
    for i in range(n_circles):
        T[i, (i + 1) % n_circles] = 1.0
        T[i, (i - 1) % n_circles] = 1.0

    # Long-range shortcut connections
    shortcuts = [
        (0, 5), (0, 7),
        (1, 6), (1, 8),
        (2, 9),
        (3, 7),
        (4, 8), (4, 1),
        (5, 2),
        (6, 0), (6, 3),
        (7, 4),
        (8, 1), (8, 5),
        (9, 3), (9, 6),
    ]
    for i, j in shortcuts:
        T[i, j] += rng.uniform(0.3, 0.8)

    # Zero out diagonal (no self-transitions) and normalise rows
    np.fill_diagonal(T, 0.0)
    T /= T.sum(axis=1, keepdims=True)
    return T


# ---------------------------------------------------------------------------
# Time-series generation
# ---------------------------------------------------------------------------

def generate_time_series(
    n_steps=100000,
    n_circles=10,
    ambient_dim=20,
    radius_mean=20.0,
    radius_std=2.0,
    noise_std=0.1,
    dwell_mean=400,
    dwell_std=100,
    min_period=40,
    max_period=400,
    subspace_dim=None,
    seed=42,
):
    """
    Generate a syllable-based time series on multiple circles.

    Each "syllable" is one visit to a circle:
      - Enter at the circle's fixed entry angle.
      - Each circle has a fixed angular velocity  ω_i = 2π / period_i,
        where period_i (steps per revolution) is linearly spaced from
        ``min_period`` (fastest) to ``max_period`` (slowest).
      - The number of complete revolutions K is drawn so that
        K × period_i ≈ dwell_mean, giving ~400 steps per visit.
        This ensures fixed entry/exit angles and fixed ω within a state.
      - On exit, choose the next circle via the off-diagonal transition
        matrix.

    Parameters
    ----------
    n_steps : int
        Total length of the time series.
    n_circles : int
        Number of circles.
    ambient_dim : int
        Dimension of the ambient space.
    radius_mean : float
        Mean radius for all circles.
    radius_std : float
        Std-dev of the radius distribution.
    noise_std : float
        Std-dev of isotropic Gaussian observation noise.
    dwell_mean : float
        Target mean dwell time per visit (~400 steps).
    dwell_std : float
        Target std-dev of dwell times (~100 steps).
    min_period : int
        Steps per revolution for the fastest circle.
    max_period : int
        Steps per revolution for the slowest circle.
    subspace_dim : int or None
        Dimension of the subspace from which circle planes are drawn.
        Controls geometric overlap (see ``create_circles`` docstring).
        None or ambient_dim = no overlap; 2 = maximum overlap.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray (n_steps, ambient_dim)   – observed positions
    states : ndarray (n_steps,)          – active circle index
    thetas : ndarray (n_steps,)          – angle on active circle
    T : ndarray (n_circles, n_circles)   – transition matrix
    radii : ndarray (n_circles,)         – circle radii
    entry_angles : ndarray (n_circles,)  – fixed entry angle per circle
    periods : ndarray (n_circles,)       – steps per revolution
    """
    rng = np.random.default_rng(seed)

    radii, planes, centers = create_circles(
        n_circles, ambient_dim, radius_mean, radius_std,
        subspace_dim=subspace_dim, seed=seed,
    )
    T = create_sparse_transition_matrix(n_circles, seed)

    # Per-circle period (steps per revolution): fast → slow
    periods = np.linspace(min_period, max_period, n_circles).astype(int)

    # Fixed angular velocity per circle (constant within a state)
    angular_velocities = 2 * np.pi / periods

    # Fixed entry angle for each circle
    entry_angles = rng.uniform(0, 2 * np.pi, n_circles)

    # Allocate output arrays
    X = np.zeros((n_steps, ambient_dim))
    states = np.zeros(n_steps, dtype=int)
    thetas = np.zeros(n_steps)

    t = 0
    current_circle = 0

    while t < n_steps:
        period = periods[current_circle]
        omega = angular_velocities[current_circle]
        entry = entry_angles[current_circle]

        # Draw target dwell, then quantise to whole revolutions
        target_dwell = rng.normal(dwell_mean, dwell_std)
        n_revs = max(1, round(target_dwell / period))
        dwell = n_revs * period                   # exact multiple → same exit angle
        dwell = min(dwell, n_steps - t)            # don't overshoot

        for s in range(dwell):
            angle = entry + s * omega
            states[t] = current_circle
            thetas[t] = angle % (2 * np.pi)

            pos = point_on_circle(
                thetas[t],
                radii[current_circle],
                planes[current_circle],
                centers[current_circle],
            )
            X[t] = pos + rng.normal(0, noise_std, ambient_dim)
            t += 1
            if t >= n_steps:
                break

        # Transition to next circle (off-diagonal only)
        current_circle = rng.choice(n_circles, p=T[current_circle])

    return X, states, thetas, T, radii, entry_angles, periods


# ---------------------------------------------------------------------------
# Data saving
# ---------------------------------------------------------------------------

def save_dataset(out_dir, X, states, thetas, T, radii, entry_angles, periods,
                 config):
    """
    Save the generated time series to disk.

    Files written
    -------------
    {out_dir}/data.npz
        Arrays: X (n_steps, ambient_dim), states (n_steps,),
        thetas (n_steps,), transition_matrix (n_circles, n_circles),
        radii (n_circles,), entry_angles (n_circles,),
        periods (n_circles,).
    {out_dir}/config.json
        Generation parameters (seed, dimensions, dwell stats, etc.)
        so the dataset is fully reproducible.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / 'data.npz',
        X=X,
        states=states,
        thetas=thetas,
        transition_matrix=T,
        radii=radii,
        entry_angles=entry_angles,
        periods=periods,
    )

    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved dataset to {out}/  "
          f"(data.npz: {X.nbytes / 1e6:.1f} MB, config.json)")


# ---------------------------------------------------------------------------
# Visualization & summary
# ---------------------------------------------------------------------------

# Default generation parameters (single source of truth)
#
# Noise level: noise_std ≈ 2.83.
# Signal power per dimension ≈ radius_mean² / ambient_dim = 400/20 = 20.
# SNR = signal_power / noise_std² = 20 / 8 ≈ 2.5.
#
# subspace_dim: controls geometric overlap between circle planes.
#   None or 20 → planes drawn from full 20D (minimal overlap, default).
#   3–6        → significant overlap, trajectories cross with noise.
#   2          → all circles coplanar (maximum overlap).
DEFAULT_CONFIG = dict(
    n_steps=100000,
    n_circles=10,
    ambient_dim=20,
    radius_mean=20.0,
    radius_std=2.0,
    noise_std=2.83,
    dwell_mean=400,
    dwell_std=100,
    min_period=40,
    max_period=400,
    subspace_dim=None,
    seed=42,
)


def main(run_umap=True):
    # ---- generate data ----
    config = DEFAULT_CONFIG.copy()
    X, states, thetas, T, radii, entry_angles, periods = generate_time_series(
        **config
    )

    # ---- save dataset ----
    save_dataset('data', X, states, thetas, T, radii, entry_angles, periods,
                 config)

    # ---- UMAP embedding (optional, on a subsample for speed) ----
    umap_max_points = 10000
    if run_umap:
        if len(X) > umap_max_points:
            idx = np.random.default_rng(42).choice(
                len(X), umap_max_points, replace=False)
            idx.sort()
            X_sub, states_sub = X[idx], states[idx]
            print(f"Computing UMAP on {umap_max_points} / {len(X)} points...")
        else:
            X_sub, states_sub = X, states
            idx = np.arange(len(X))
            print("Computing UMAP embedding...")
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, metric='euclidean',
                            random_state=42)
        X_umap = reducer.fit_transform(X_sub)

    # ---- figure ----
    if run_umap:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) State sequence
    ax = fig.add_subplot(gs[0, 0]) if run_umap else axes[0]
    ax.scatter(np.arange(len(states)), states, c=states, cmap='tab10',
               s=1, alpha=0.4)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Circle index')
    ax.set_title('Markov State Sequence')
    ax.set_yticks(range(10))
    ax.grid(True, alpha=0.3)

    # (b) Transition matrix
    ax = fig.add_subplot(gs[0, 1]) if run_umap else axes[1]
    im = ax.imshow(T, cmap='Blues', vmin=0)
    ax.set_xlabel('To circle')
    ax.set_ylabel('From circle')
    ax.set_title('Sparse Transition Matrix')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] > 0.005:
                ax.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if T[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) Period (steps per revolution) per circle
    ax = fig.add_subplot(gs[0, 2]) if run_umap else axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ax.bar(range(10), periods, color=colors)
    ax.set_xlabel('Circle index')
    ax.set_ylabel('Steps per revolution')
    ax.set_title('Period  (fast → 10× slower)')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')

    if run_umap:
        # (d) UMAP coloured by circle state (bottom-left, wide)
        ax = fig.add_subplot(gs[1, 0:2])
        sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=states_sub,
                        cmap='tab10', s=3, alpha=0.5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Embedding (coloured by circle)')
        cbar = plt.colorbar(sc, ax=ax, label='Circle index')
        cbar.set_ticks(range(10))
        ax.grid(True, alpha=0.2)

        # (e) UMAP coloured by time (bottom-right)
        ax = fig.add_subplot(gs[1, 2])
        sc2 = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=idx,
                         cmap='viridis', s=3, alpha=0.5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Embedding (coloured by time)')
        plt.colorbar(sc2, ax=ax, label='Time step')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('markov_circles_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Standalone UMAP figure ----
    if run_umap:
        from estimate_dimension import levina_bickel_estimator

        sdim = config.get('subspace_dim')
        if sdim is None:
            sdim = config['ambient_dim']

        # Levina-Bickel on the 2D UMAP embedding at several k values
        lb_ks = [10, 30, 100]
        lb_results = {}
        print("Computing Levina-Bickel on UMAP embedding...")
        for k in lb_ks:
            m_hat, _ = levina_bickel_estimator(X_umap, k)
            lb_results[k] = m_hat
        lb_str = '  '.join(f'k={k}: {lb_results[k]:.2f}' for k in lb_ks)
        print(f"  LB dimension in UMAP space:  {lb_str}")

        title = (f'UMAP — {config["n_circles"]} circles in '
                 f'{config["ambient_dim"]}D  '
                 f'(subspace dim = {sdim})')
        subtitle = f'Levina-Bickel dim in UMAP space:  {lb_str}'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sc1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=states_sub,
                          cmap='tab10', s=3, alpha=0.5)
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Coloured by circle')
        cbar1 = plt.colorbar(sc1, ax=ax1, label='Circle index')
        cbar1.set_ticks(range(config['n_circles']))
        ax1.grid(True, alpha=0.2)

        sc2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=idx,
                          cmap='viridis', s=3, alpha=0.5)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('Coloured by time')
        plt.colorbar(sc2, ax=ax2, label='Time step')
        ax2.grid(True, alpha=0.2)

        fig.suptitle(title + '\n' + subtitle, fontsize=13, fontweight='bold',
                     y=1.04)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f'umap_subspace_{sdim}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")

    # ---- Sample data images: raw dimensions vs time ----
    # Evenly spaced windows: beginning, middle, end of the time series
    n_samples = 3
    window = 3000  # time steps per sample
    total = len(states)
    starts = [0, total // 2 - window // 2, total - window]

    for idx, t0 in enumerate(starts):
        t1 = t0 + window

        fig = plt.figure(figsize=(16, 6))
        # 2 rows, 2 cols: left column for plots, right column for colorbar
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4],
                              width_ratios=[1, 0.02], hspace=0.05, wspace=0.03)

        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_cbar = fig.add_subplot(gs[1, 1])

        # Top strip: state sequence coloured by circle
        for t in range(t0, t1):
            ax_top.axvspan(t, t + 1, color=plt.cm.tab10(states[t] / 10),
                           alpha=0.8, linewidth=0)
        ax_top.set_xlim(t0, t1)
        ax_top.set_yticks([])
        ax_top.set_ylabel('State', fontsize=10)
        ax_top.set_title(f'Sample window  t = {t0} … {t1}  '
                         f'({window} steps)', fontsize=12)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Bottom: heatmap of all dimensions
        chunk = X[t0:t1, :].T           # (ambient_dim, window)
        im = ax_bot.imshow(chunk, aspect='auto', cmap='RdBu_r',
                           extent=[t0, t1, chunk.shape[0] - 0.5, -0.5],
                           interpolation='none')
        ax_bot.set_xlabel('Time step', fontsize=10)
        ax_bot.set_ylabel('Dimension', fontsize=10)
        ax_bot.set_yticks(range(chunk.shape[0]))

        # Colorbar in its own aligned column
        plt.colorbar(im, cax=ax_cbar, label='Value')

        # Hide the top-right cell so it doesn't take space
        ax_dummy = fig.add_subplot(gs[0, 1])
        ax_dummy.axis('off')

        fname = f'sample_window_{idx}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}  (t={t0}..{t1})")

    # ---- console summary ----
    print("Saved plot to markov_circles_timeseries.png")
    print(f"\nData shape : {X.shape}")
    print(f"Ambient dim: {X.shape[1]}")
    print(f"Time steps : {X.shape[0]}")

    # Compute per-circle dwell-time statistics from the state sequence
    dwell_lengths = {i: [] for i in range(10)}
    run_start = 0
    for t in range(1, len(states)):
        if states[t] != states[t - 1]:
            dwell_lengths[states[t - 1]].append(t - run_start)
            run_start = t
    dwell_lengths[states[-1]].append(len(states) - run_start)  # final run

    print(f"\n{'Circle':<8} {'Radius':<8} {'Period':<8} {'Steps':<8} "
          f"{'Frac%':<7} {'#Visits':<8} {'MeanDwell':<10} {'StdDwell':<10}")
    print('-' * 75)
    for i in range(10):
        n_steps_i = (states == i).sum()
        frac = 100 * n_steps_i / len(states)
        dwells = dwell_lengths[i]
        n_vis = len(dwells)
        m_dw = np.mean(dwells) if dwells else 0
        s_dw = np.std(dwells) if dwells else 0
        print(f"{i:<8} {radii[i]:<8.2f} {periods[i]:<8} "
              f"{n_steps_i:<8} {frac:<7.1f} {n_vis:<8} {m_dw:<10.1f} {s_dw:<10.1f}")

    print(f"\n{'Entry angles (rad):'}")
    for i in range(10):
        print(f"  Circle {i}: {entry_angles[i]:.3f}")

    print(f"\nTransition matrix (off-diagonal, non-zero entries):")
    for i in range(T.shape[0]):
        nonzero = [(j, T[i, j]) for j in range(T.shape[1]) if T[i, j] > 1e-6]
        entries = ', '.join(f'{j}:{p:.3f}' for j, p in nonzero)
        print(f"  Circle {i} → {entries}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate a Markov-switching time series on circles.')
    parser.add_argument('--no-umap', action='store_true',
                        help='Skip the UMAP embedding (much faster)')
    parser.add_argument('--subspace-dim', type=int, default=None,
                        help='Dimension of the subspace from which circle '
                             'planes are drawn.  Controls geometric overlap: '
                             '20 (default) = no overlap, 4 = significant, '
                             '2 = maximum.  Range: 2..ambient_dim.')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Total time steps to generate '
                             '(default: 100000)')
    args = parser.parse_args()
    if args.subspace_dim is not None:
        DEFAULT_CONFIG['subspace_dim'] = args.subspace_dim
    if args.n_steps is not None:
        DEFAULT_CONFIG['n_steps'] = args.n_steps
    main(run_umap=not args.no_umap)
