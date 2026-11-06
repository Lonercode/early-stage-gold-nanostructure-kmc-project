# export_sweep_tables.py
# Recomputes all metrics fresh from sweep_summary.npy and writes:
#   results/sweep_results.csv         — per-temperature summary (with N_111_std, N_100_std)
#   results/sweep_results_by_seed.csv — per-seed breakdown (all 21 runs)
#
import numpy as np
import csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordination import compute_all_cn
from ase.io import read

SWEEP   = os.path.join(os.path.dirname(__file__), '..', 'sweeps', 'sweep_summary.npy')
OUT     = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)


def pca_geo_ar(positions):
    """L1 / mean(L2, L3); axes are 2-sigma spreads from PCA on centred positions."""
    p    = positions - positions.mean(axis=0)
    vals = np.linalg.eigvalsh(np.cov(p.T))[::-1]   # descending eigenvalues
    L    = 2.0 * np.sqrt(np.maximum(vals, 0))
    return float(L[0] / np.mean(L[1:]))


def cn_counts(cn_array):
    n111 = int((cn_array == 9).sum())
    n100 = int((cn_array == 8).sum())
    proxy_ar = n100 / n111 if n111 > 0 else float('nan')
    return n111, n100, proxy_ar


# ── load raw records ──────────────────────────────────────────────────────────
records = list(np.load(SWEEP, allow_pickle=True))
records.sort(key=lambda r: (r['T_K'], r['rng_seed']))
temperatures = sorted(set(r['T_K'] for r in records))

# ── recompute every metric from raw arrays ────────────────────────────────────
for r in records:
    r['_geo_ar']              = pca_geo_ar(r['positions'])
    r['_n111'], r['_n100'], r['_proxy_ar'] = cn_counts(r['cn'])
    r['_dep_diff'] = r['n_dep'] / r['n_diff'] if r['n_diff'] > 0 else float('nan')

# ── 1. per-seed CSV ───────────────────────────────────────────────────────────
by_seed_path = os.path.join(OUT, 'sweep_results_by_seed.csv')
by_seed_cols = ['T_K', 'seed', 'N_111', 'N_100', 'proxy_AR', 'geo_AR',
                'n_dep', 'n_diff', 'dep_diff_ratio', 'n_steps']
with open(by_seed_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(by_seed_cols)
    for r in records:
        w.writerow([
            int(r['T_K']),
            int(r['rng_seed']),
            r['_n111'],
            r['_n100'],
            f"{r['_proxy_ar']:.4f}",
            f"{r['_geo_ar']:.4f}",
            int(r['n_dep']),
            int(r['n_diff']),
            f"{r['_dep_diff']:.4f}",
            len(r['event_log']),
        ])

# ── 2. per-temperature summary CSV ───────────────────────────────────────────
summary_path = os.path.join(OUT, 'sweep_results.csv')
summary_cols = [
    'T_K',
    'geo_AR_mean', 'geo_AR_std',
    'N_111_mean',  'N_111_std',
    'N_100_mean',  'N_100_std',
    'proxy_AR_mean', 'proxy_AR_std',
    'dep_diff_ratio_mean',
]
with open(summary_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(summary_cols)
    for T in temperatures:
        recs = [r for r in records if r['T_K'] == T]
        geo  = [r['_geo_ar']    for r in recs]
        n111 = [r['_n111']      for r in recs]
        n100 = [r['_n100']      for r in recs]
        pxy  = [r['_proxy_ar']  for r in recs]
        dd   = [r['_dep_diff']  for r in recs]
        w.writerow([
            int(T),
            f'{np.mean(geo):.4f}',  f'{np.std(geo):.4f}',
            f'{np.mean(n111):.1f}', f'{np.std(n111):.1f}',
            f'{np.mean(n100):.1f}', f'{np.std(n100):.1f}',
            f'{np.mean(pxy):.4f}',  f'{np.std(pxy):.4f}',
            f'{np.mean(dd):.4f}',
        ])

# ── print both tables ─────────────────────────────────────────────────────────
print("=" * 72)
print("sweep_results_by_seed.csv  (21 rows, all values recomputed from raw arrays)")
print("=" * 72)
print(f"{'T_K':>5}  {'seed':>5}  {'N_111':>6}  {'N_100':>6}  "
      f"{'proxy_AR':>9}  {'geo_AR':>7}  {'n_dep':>6}  {'n_diff':>7}  "
      f"{'dep/diff':>9}  {'n_steps':>8}")
print('-' * 72)
for r in records:
    print(f"{int(r['T_K']):>5}  {int(r['rng_seed']):>5}  {r['_n111']:>6}  "
          f"{r['_n100']:>6}  {r['_proxy_ar']:>9.4f}  {r['_geo_ar']:>7.4f}  "
          f"{int(r['n_dep']):>6}  {int(r['n_diff']):>7}  "
          f"{r['_dep_diff']:>9.4f}  {len(r['event_log']):>8}")

print()
print("=" * 80)
print("sweep_results.csv  (7 rows, per-temperature summary)")
print("=" * 80)
print(f"{'T_K':>5}  {'geo_AR':>14}  {'N_111':>14}  {'N_100':>14}  "
      f"{'proxy_AR':>14}  {'dep/diff':>9}")
print(f"{'':>5}  {'mean±std':>14}  {'mean±std':>14}  {'mean±std':>14}  "
      f"{'mean±std':>14}  {'mean':>9}")
print('-' * 80)
for T in temperatures:
    recs = [r for r in records if r['T_K'] == T]
    geo  = [r['_geo_ar']   for r in recs]
    n111 = [r['_n111']     for r in recs]
    n100 = [r['_n100']     for r in recs]
    pxy  = [r['_proxy_ar'] for r in recs]
    dd   = [r['_dep_diff'] for r in recs]
    print(f"{int(T):>5}  "
          f"{np.mean(geo):>6.4f}±{np.std(geo):<6.4f}  "
          f"{np.mean(n111):>6.1f}±{np.std(n111):<6.1f}  "
          f"{np.mean(n100):>6.1f}±{np.std(n100):<6.1f}  "
          f"{np.mean(pxy):>6.4f}±{np.std(pxy):<6.4f}  "
          f"{np.mean(dd):>9.4f}")

print(f"\nFiles written:")
print(f"  {summary_path}")
print(f"  {by_seed_path}")
