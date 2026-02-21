"""
PyTorch Dataset for the synthetic-song time series.

Loads data saved by markov_circles_timeseries.py and serves fixed-length
windows, ready for a BERT-style masked prediction model.

Usage
-----
    from dataset import SyntheticSongDataset

    ds = SyntheticSongDataset('data', seq_len=512)
    x, state, mask = ds[0]
    # x     : (seq_len, ambient_dim)  – observed positions
    # state : (seq_len,)              – circle index labels
    # mask  : (seq_len,)              – True where masked (to be predicted)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticSongDataset(Dataset):
    """
    Sliding-window dataset over the synthetic-song time series.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``data.npz`` and ``config.json``
        (written by ``markov_circles_timeseries.save_dataset``).
    seq_len : int
        Length of each window (number of time steps).
    stride : int or None
        Step between consecutive windows.  Defaults to ``seq_len``
        (non-overlapping).  Set to 1 for maximum overlap.
    mask_ratio : float
        Fraction of time steps to mask per window (0.0–1.0).
    mask_patch_size : int
        Contiguous patch size for each masked region (used when
        mask_patch_min/max are not set).
    mask_patch_min : int or None
        Minimum patch size for variable-size masking.  If None,
        defaults to mask_patch_size (fixed-size patches).
    mask_patch_max : int or None
        Maximum patch size for variable-size masking.  If None,
        defaults to mask_patch_size (fixed-size patches).
        Set both min and max for stochastic patch sizes, e.g.
        mask_patch_min=8, mask_patch_max=64.
    mask_seed : int or None
        If set, masking is deterministic for reproducibility.
        If None, masking is random each time ``__getitem__`` is called.
    """

    def __init__(
        self,
        data_dir='data',
        seq_len=512,
        stride=None,
        mask_ratio=0.15,
        mask_patch_size=16,
        mask_patch_min=None,
        mask_patch_max=None,
        mask_seed=None,
    ):
        data_dir = Path(data_dir)

        # Load arrays
        npz = np.load(data_dir / 'data.npz')
        self.X = npz['X'].astype(np.float32)           # (n_steps, ambient_dim)
        self.states = npz['states'].astype(np.int64)    # (n_steps,)
        self.thetas = npz['thetas'].astype(np.float32)  # (n_steps,)

        # Load generation config
        with open(data_dir / 'config.json') as f:
            self.config = json.load(f)

        self.n_steps, self.ambient_dim = self.X.shape
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        # Variable patch sizes: defaults to fixed size if min/max not given
        self.mask_patch_min = mask_patch_min if mask_patch_min is not None else mask_patch_size
        self.mask_patch_max = mask_patch_max if mask_patch_max is not None else mask_patch_size
        self.mask_seed = mask_seed

        # Pre-compute window start indices
        self.starts = list(range(
            0, self.n_steps - self.seq_len + 1, self.stride
        ))

    def __len__(self):
        return len(self.starts)

    def _generate_mask(self, idx):
        """
        Generate a patch-based mask for one window.

        Each patch has a stochastic size drawn uniformly from
        [mask_patch_min, mask_patch_max].  Patches are placed until
        the target mask_ratio is approximately reached.

        Returns a boolean array of shape (seq_len,) where True means
        "this time step is masked and should be predicted."
        """
        if self.mask_seed is not None:
            rng = np.random.default_rng(self.mask_seed + idx)
        else:
            rng = np.random.default_rng()

        mask = np.zeros(self.seq_len, dtype=bool)
        target_masked = int(self.mask_ratio * self.seq_len)

        # Use variable-size patches if min/max range is set
        p_min = self.mask_patch_min
        p_max = self.mask_patch_max

        total_masked = 0
        attempts = 0
        max_attempts = self.seq_len  # safety limit

        while total_masked < target_masked and attempts < max_attempts:
            patch_size = rng.integers(p_min, p_max + 1)  # inclusive
            max_start = self.seq_len - patch_size
            if max_start <= 0:
                mask[:] = True
                return mask

            s = rng.integers(0, max_start + 1)
            new_masked = (~mask[s : s + patch_size]).sum()
            mask[s : s + patch_size] = True
            total_masked += new_masked
            attempts += 1

        return mask

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        t1 = t0 + self.seq_len

        x = torch.from_numpy(self.X[t0:t1].copy())
        state = torch.from_numpy(self.states[t0:t1].copy())
        mask = torch.from_numpy(self._generate_mask(idx))

        return x, state, mask

    @property
    def feature_dim(self):
        """Dimension of each time step (ambient_dim)."""
        return self.ambient_dim

    @property
    def n_classes(self):
        """Number of circle states."""
        return self.config['n_circles']


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ds = SyntheticSongDataset('data', seq_len=512, stride=256,
                              mask_ratio=0.15, mask_patch_size=16,
                              mask_seed=0)
    print(f"Dataset: {len(ds)} windows")
    print(f"  seq_len      = {ds.seq_len}")
    print(f"  feature_dim  = {ds.feature_dim}")
    print(f"  n_classes    = {ds.n_classes}")
    print(f"  mask_ratio   = {ds.mask_ratio}")
    print(f"  patch_size   = {ds.mask_patch_size}")

    x, state, mask = ds[0]
    print(f"\nSample window [0]:")
    print(f"  x.shape      = {x.shape}")
    print(f"  state.shape  = {state.shape}")
    print(f"  mask.shape   = {mask.shape}")
    print(f"  masked steps = {mask.sum().item()} / {ds.seq_len} "
          f"({100 * mask.float().mean():.1f}%)")
    print(f"  x dtype      = {x.dtype}")
    print(f"  x range      = [{x.min():.2f}, {x.max():.2f}]")
