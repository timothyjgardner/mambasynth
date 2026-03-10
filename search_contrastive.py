"""Automated hyperparameter search for contrastive loss parameters.

Sweeps contrastive-lambda and contrastive-temp, trains each configuration,
evaluates silhouette scores, and reports the best result.
"""

import subprocess
import os
import re
import csv
import sys
import time
from itertools import product

BASE_TRAIN = [
    sys.executable, 'masked_model_gpu_mamba_next.py',
    '--d-state', '32', '--seq-len', '1024', '--stride', '256',
    '--no-compile', '--no-train-eval', '--max-horizon', '1', '--epochs', '200',
]

BASE_EVAL = [sys.executable, 'evaluate_representations.py']

LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
TEMP_VALUES = [0.2, 0.3, 0.5, 1.0]

LOG_FILE = 'contrastive_search_results_2.csv'


def parse_silhouette(output):
    """Extract silhouette scores from evaluate_representations.py output."""
    scores = {}
    in_sil = False
    for line in output.splitlines():
        if 'Silhouette scores' in line:
            in_sil = True
            continue
        if in_sil:
            m = re.match(r'\s+(Input|Layer \d+|Output):\s*([-\d.]+)', line)
            if m:
                scores[m.group(1)] = float(m.group(2))
            elif line.strip() == '':
                break
    return scores


def best_intermediate_sil(scores):
    """Mean silhouette across intermediate layers (Layer 2–6)."""
    vals = [scores[k] for k in scores
            if k.startswith('Layer') and k not in ('Layer 1', 'Layer 7')]
    return sum(vals) / len(vals) if vals else -1.0


def run_config(lam, temp, run_idx, total):
    tag = f"lam{lam}_temp{temp}"
    ckpt = f"model_search_{tag}.pt"

    print(f"\n{'='*60}")
    print(f"[{run_idx}/{total}]  λ={lam}  τ={temp}")
    print(f"{'='*60}")

    t0 = time.time()
    train_cmd = BASE_TRAIN + [
        '--contrastive-lambda', str(lam),
        '--contrastive-temp', str(temp),
        '--checkpoint', ckpt,
    ]
    r = subprocess.run(train_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  TRAIN FAILED: {r.stderr[-500:]}")
        return None
    train_time = time.time() - t0

    val_mse_match = re.search(r'Best val MSE:\s*([\d.]+)', r.stdout)
    val_mse = float(val_mse_match.group(1)) if val_mse_match else -1.0
    print(f"  Train: {train_time:.0f}s  val_mse={val_mse:.4f}")

    eval_cmd = BASE_EVAL + ['--checkpoint', ckpt, '--no-lb']
    r2 = subprocess.run(eval_cmd, capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"  EVAL FAILED: {r2.stderr[-500:]}")
        return None

    scores = parse_silhouette(r2.stdout)
    mid_sil = best_intermediate_sil(scores)
    print(f"  Silhouettes: {scores}")
    print(f"  Mean intermediate sil (L2-6): {mid_sil:.3f}")

    for src in ('representation_umap.png', 'training_loss.png'):
        if os.path.exists(src):
            base, ext = os.path.splitext(src)
            dst = f"{base}_{tag}{ext}"
            os.rename(src, dst)
            print(f"  {src} -> {dst}")

    return {
        'lambda': lam, 'temp': temp, 'val_mse': val_mse,
        'mid_sil': mid_sil, 'train_time': train_time,
        **{f'sil_{k}': v for k, v in scores.items()},
    }


def main():
    configs = list(product(LAMBDA_VALUES, TEMP_VALUES))
    print(f"Searching {len(configs)} configurations: "
          f"{len(LAMBDA_VALUES)} λ × {len(TEMP_VALUES)} τ")

    results = []
    with open(LOG_FILE, 'w', newline='') as f:
        writer = None
        for i, (lam, temp) in enumerate(configs, 1):
            row = run_config(lam, temp, i, len(configs))
            if row is None:
                continue
            results.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    if not results:
        print("No successful runs!")
        return

    best = max(results, key=lambda r: r['mid_sil'])
    print(f"\n{'='*60}")
    print(f"BEST: λ={best['lambda']}  τ={best['temp']}  "
          f"mid_sil={best['mid_sil']:.3f}  val_mse={best['val_mse']:.4f}")
    print(f"{'='*60}")
    print(f"\nFull results saved to {LOG_FILE}")


if __name__ == '__main__':
    main()
