# check_pca_seed_level.py
# Prints seed-level PCA geometric AR for every (T, seed) in sweep_summary.npy,
# plus per-temperature mean/std and axis-consistency angles between seeds.
# Used to confirm the 340K inter-seed spread and identify which seed drives the
# high/low ends of the distribution.

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordination import compute_all_cn
from ase.io import read

SWEEP   = os.path.join(os.path.dirname(__file__), '..', 'sweeps', 'sweep_summary.npy')
SEED_XY = os.path.join(os.path.dirname(__file__), '..', 'octahedron_4nm_relaxed.xyz')


def pca_geo_ar(positions):
    """L1 / mean(L2, L3) from PCA; also returns (L1, L2, L3) and eigenvectors."""
    p    = positions - positions.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(p.T))
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]
    L    = 2.0 * np.sqrt(np.maximum(vals, 0))
    return L[0] / np.mean(L[1:]), L, vecs[:, 0]


def cn_counts(cn):
    return int((cn == 9).sum()), int((cn == 8).sum())


# ── seed reference ────────────────────────────────────────────────────────────
seed_pos = read(SEED_XY).get_positions().astype(float)
seed_cn  = compute_all_cn(seed_pos)
seed_ar, seed_L, _ = pca_geo_ar(seed_pos)
seed_n111, seed_n100 = cn_counts(seed_cn)
print(f"Seed: geo_AR={seed_ar:.4f}  N(111)={seed_n111}  N(100)={seed_n100}")
print(f"      PCA lengths: {seed_L[0]:.2f}, {seed_L[1]:.2f}, {seed_L[2]:.2f} Å\n")

# ── load records ──────────────────────────────────────────────────────────────
records = list(np.load(SWEEP, allow_pickle=True))
records.sort(key=lambda r: (r['T_K'], r['rng_seed']))
temperatures = sorted(set(r['T_K'] for r in records))

# ── full seed-level table ─────────────────────────────────────────────────────
print(f"{'T':>4} {'sd':>4}  "
      f"{'L1':>6} {'L2':>6} {'L3':>6}  "
      f"{'geo_AR':>7}  {'L1/L2':>6} {'L1/L3':>6}  "
      f"{'N(111)':>7} {'N(100)':>7}")
print('-' * 72)

results = []
for r in records:
    g, L, axis = pca_geo_ar(r['positions'])
    n111, n100 = cn_counts(r['cn'])
    results.append({'T': r['T_K'], 'seed': r['rng_seed'],
                    'geo_ar': g, 'L': L, 'axis': axis,
                    'n111': n111, 'n100': n100})
    print(f"{int(r['T_K']):>4} {r['rng_seed']:>4}  "
          f"{L[0]:>6.2f} {L[1]:>6.2f} {L[2]:>6.2f}  "
          f"{g:>7.4f}  {L[0]/L[1]:>6.3f} {L[0]/L[2]:>6.3f}  "
          f"{n111:>7} {n100:>7}")

# ── per-temperature summary ───────────────────────────────────────────────────
print(f"\nPer-temperature summary (geo_AR):")
print(f"{'T':>4}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  "
      f"{'hi seed':>8}  {'lo seed':>8}")
print('-' * 60)
for T in temperatures:
    T_res = [res for res in results if res['T'] == T]
    vals  = [(res['geo_ar'], res['seed']) for res in T_res]
    ars   = [v[0] for v in vals]
    hi    = max(vals, key=lambda x: x[0])
    lo    = min(vals, key=lambda x: x[0])
    print(f"{int(T):>4}  {np.mean(ars):>7.4f}  {np.std(ars):>7.4f}  "
          f"{lo[0]:>7.4f}  {hi[0]:>7.4f}  "
          f"s={hi[1]:>4}     s={lo[1]:>4}")

# ── axis-consistency angles ───────────────────────────────────────────────────
print(f"\nAxis-consistency: pairwise angles between principal axes (degrees)")
print(f"{'T':>4}  {'s7·s42':>8}  {'s7·s123':>9}  {'s42·s123':>10}")
print('-' * 38)
for T in temperatures:
    T_res = sorted([res for res in results if res['T'] == T], key=lambda x: x['seed'])
    axes  = [res['axis'] for res in T_res]
    def ang(a, b):
        return np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))
    print(f"{int(T):>4}  {ang(axes[0],axes[1]):>8.1f}  "
          f"{ang(axes[0],axes[2]):>9.1f}  {ang(axes[1],axes[2]):>10.1f}")
