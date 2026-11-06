# au_nanorod_sweep_silver_500.py — targeted elongation test (CTAB + silver, 500 dep)
#
# Upload to Kaggle alongside:
#   octahedron_4nm_relaxed.xyz
#   coordination.py  diffusion.py  deposition.py  neighbors.py  kmc.py  silver.py
#
# Same as au_nanorod_sweep_silver.py but with:
#   TARGET_DEP = 500 (vs 100)
#   TEMPERATURES = [280, 300] — low and mid anchor temperatures only
#   Outputs: sweep_silver_500_summary.npy / sweep_silver_500_results.csv
#   Checkpoints / final XYZ filenames include _500 suffix → no collision with
#   Phase 2 (target=100) files
#
# 340K excluded from this run: at target=500 it is estimated 14–19 hrs (3 seeds)
# and would bust Kaggle's 12-hour session limit. Run separately if needed after
# reviewing 280K and 300K results.
#
# Wall-time estimate (Phase 1 measured: 280K ≈ 300 s/seed at target=100):
#   280K × 500 dep: 5 × ~300 s/seed × 3 seeds ≈ 1.25 hrs
#   300K × 500 dep: 5 × ~500 s/seed × 3 seeds ≈ 2.1 hrs
#   Total (2T × 3 seeds = 6 runs): ~3.3 hrs — safely within Kaggle's 12-hour limit.
#
# N_STEPS_CAP raised to 2_000_000 (vs 500_000 in Phase 2) as a precaution —
# at target=500 and lower dep%, step counts could approach the old cap at high T.

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ase', '-q'])

import csv, os, time
import numpy as np
from ase.io import read, write as ase_write
from ase import Atoms as AseAtoms

from coordination import compute_all_cn
from kmc import run_kmc
from silver import build_silver_mask, print_silver_report

# ── Sweep parameters ──────────────────────────────────────────────────────────
SEED_FILE        = 'octahedron_4nm_relaxed.xyz'
K_DEP            = 3.5e7    # s⁻¹; same as Phase 1 and Phase 2 — fixed solution-side Au flux
TARGET_DEP       = 500      # deposition events per run — 5× Phase 2 for elongation test
N_STEPS_CAP      = 2_000_000  # raised from 500k; precaution for higher step counts at 500 dep
CHECKPOINT_EVERY = 500      # write xyz frame every N steps

SILVER_RNG_SEED = 42        # selects which {111}–{111} ridge silver stabilises;
                             # fixed across all runs so T is the only variable

TEMPERATURES = [280, 300]   # low and mid anchor temperatures; 340K run separately
RNG_SEEDS    = [42, 7, 123] # 3 Gillespie seeds per T

SUMMARY_FILE = 'sweep_silver_500_summary.npy'
CSV_FILE     = 'sweep_silver_500_results.csv'

CSV_FIELDS = [
    'T_K', 'rng_seed', 'silver_rng_seed', 'ridge_idx', 'n_proximal',
    'geo_AR', 'facet_AR',
    'n_dep', 'n_diff', 'dep_pct', 'final_N',
    'kmc_time_s', 'wall_s', 'step_rate_s',
]


# ── Geometric AR via PCA ──────────────────────────────────────────────────────
def pca_geo_ar(positions: np.ndarray) -> tuple:
    """
    L1 / mean(L2, L3) from PCA of atom positions.

    Returns (geo_AR, (L1, L2, L3), principal_axis).
    Same formula as scripts/check_pca_seed_level.py — results are directly
    comparable to Phase 1 and Phase 2 post-hoc analysis.
    """
    p          = positions - positions.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(p.T))
    idx        = np.argsort(vals)[::-1]
    vals       = vals[idx]; vecs = vecs[:, idx]
    L          = 2.0 * np.sqrt(np.maximum(vals, 0))
    return L[0] / np.mean(L[1:]), L, vecs[:, 0]


# ── Facet AR (N_100 / N_111) — same as Phase 1 and Phase 2 sweep ─────────────
def facet_ar(cn: np.ndarray) -> float:
    """N({100}) / N({111}) surface-atom ratio; returns nan if no CN=9 atoms."""
    n_111 = int((cn == 9).sum())
    n_100 = int((cn == 8).sum())
    return n_100 / n_111 if n_111 > 0 else float('nan')


# ── Incremental saves ─────────────────────────────────────────────────────────
def save_summary(records: list) -> None:
    np.save(SUMMARY_FILE, records)


def save_csv(records: list) -> None:
    with open(CSV_FILE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


# ── Load seed ─────────────────────────────────────────────────────────────────
seed_atoms = read(SEED_FILE)
pos0       = seed_atoms.get_positions().astype(float)
seed_cn    = compute_all_cn(pos0)
seed_geo,  seed_L, _ = pca_geo_ar(pos0)
seed_far   = facet_ar(seed_cn)

print(f"Seed: {len(pos0)} atoms  |  geo_AR={seed_geo:.4f}  facet_AR={seed_far:.3f}")
print(f"  PCA lengths: {seed_L[0]:.2f}, {seed_L[1]:.2f}, {seed_L[2]:.2f} Å")
print(f"  Seed file: {SEED_FILE}\n")

# ── Silver mask — computed once at startup for the printed report.
# Note: build_silver_mask is also called inside run_kmc (via silver_rng_seed)
# for each run, so the mask is computed twice per run. The startup call here
# is purely for the log output and costs ~10 ms — acceptable redundancy.
_, ag_info = build_silver_mask(pos0, seed_cn, rng_seed=SILVER_RNG_SEED)
print_silver_report(seed_cn, ag_info)
print(f"silver_rng_seed = {SILVER_RNG_SEED}  (fixed for all runs)\n")

print(f"Temperatures : {TEMPERATURES} K  (340K deferred — run separately at target=500)")
print(f"KMC seeds    : {RNG_SEEDS}")
print(f"Target dep   : {TARGET_DEP} events per run")
print(f"k_dep        : {K_DEP:.2e} s⁻¹")
print(f"Outputs      : {SUMMARY_FILE}  {CSV_FILE}")
print(f"Est. wall    : ~3.3 hrs (2T × 3 seeds)\n")

# ── Resume support ────────────────────────────────────────────────────────────
if os.path.exists(SUMMARY_FILE):
    existing = list(np.load(SUMMARY_FILE, allow_pickle=True))
    done     = {(r['T_K'], r['rng_seed']) for r in existing}
    print(f"Resuming — {len(existing)} runs already complete: {done}\n")
else:
    existing = []
    done     = set()

# ── Sweep ─────────────────────────────────────────────────────────────────────
all_records = list(existing)

for T in TEMPERATURES:
    T_records = []

    for rng_seed in RNG_SEEDS:
        if (T, rng_seed) in done:
            print(f"  T={T}K seed={rng_seed}: already done, skipping.")
            T_records.append(next(r for r in all_records
                                  if r['T_K'] == T and r['rng_seed'] == rng_seed))
            continue

        label     = f'T{int(T)}K_s{rng_seed}_silver_500'
        ckpt_path = f'sweep_{label}.xyz'
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        print(f"  T={T} K  seed={rng_seed}  (silver_rng_seed={SILVER_RNG_SEED})")
        t_wall = time.perf_counter()

        result = run_kmc(
            pos0,
            T                 = T,
            n_steps           = N_STEPS_CAP,
            k_dep             = K_DEP,
            rng               = np.random.default_rng(rng_seed),
            debug             = False,
            checkpoint_every  = CHECKPOINT_EVERY,
            checkpoint_path   = ckpt_path,
            target_dep_events = TARGET_DEP,
            silver_rng_seed   = SILVER_RNG_SEED,
        )

        elapsed    = time.perf_counter() - t_wall
        n_steps    = len(result['event_log'])
        n_dep      = result['n_dep']
        n_diff     = result['n_diff']
        final_N    = len(result['positions'])
        kmc_time   = result['time']
        dep_pct    = n_dep / n_steps * 100
        step_rate  = elapsed / n_steps if n_steps > 0 else 0.0

        geo, L, ax = pca_geo_ar(result['positions'])
        far        = facet_ar(result['cn'])

        print(f"    steps={n_steps:,}  dep={n_dep}({dep_pct:.1f}%)  "
              f"diff={n_diff:,}  N={final_N}  "
              f"geo_AR={geo:.4f}  facet_AR={far:.3f}  "
              f"wall={elapsed:.0f}s  ({step_rate:.3f} s/step)")

        # Save final structure for this (T, seed)
        final_atoms = AseAtoms('Au' * final_N, positions=result['positions'])
        final_atoms.info.update({
            'T_K': T, 'rng_seed': rng_seed,
            'silver_rng_seed': SILVER_RNG_SEED,
            'ridge_idx': int(ag_info['selected_ridge_idx']),
            'kmc_time_s': kmc_time,
            'n_dep': n_dep, 'n_diff': n_diff,
            'k_dep': K_DEP,
            'geo_AR': geo, 'facet_AR': far,
        })
        ase_write(f'final_{label}.xyz', final_atoms, format='extxyz')

        record = {
            'T_K':             T,
            'rng_seed':        rng_seed,
            'silver_rng_seed': SILVER_RNG_SEED,
            'ridge_idx':       int(ag_info['selected_ridge_idx']),
            'n_proximal':      int(ag_info['n_proximal']),
            'geo_AR':          geo,
            'facet_AR':        far,
            'n_dep':           n_dep,
            'n_diff':          n_diff,
            'dep_pct':         dep_pct,
            'final_N':         final_N,
            'kmc_time_s':      kmc_time,
            'wall_s':          elapsed,
            'step_rate_s':     step_rate,
            # heavy arrays — same schema as Phase 2 summary
            'positions':       result['positions'].copy(),
            'cn':              result['cn'].copy(),
            'event_log':       result['event_log'],
        }
        all_records.append(record)
        T_records.append(record)

        # Incremental save after every run — crash-safe and resume-safe
        save_summary(all_records)
        save_csv([{k: r[k] for k in CSV_FIELDS} for r in all_records])
        print(f"    -> {SUMMARY_FILE}  {CSV_FILE}  updated ({len(all_records)} runs)\n")

    # Per-temperature summary across seeds
    geos  = [r['geo_AR']  for r in T_records]
    walls = [r['wall_s']  for r in T_records]
    ns    = [len(r['event_log']) for r in T_records]
    print(f"  T={T}K summary: geo_AR={np.mean(geos):.4f}±{np.std(geos):.4f}  "
          f"steps={np.mean(ns):.0f}±{np.std(ns):.0f}  "
          f"wall={np.sum(walls):.0f}s total\n")

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*76}")
print(f"SWEEP COMPLETE — {len(all_records)} runs  "
      f"(silver_rng_seed={SILVER_RNG_SEED}, ridge={ag_info['selected_ridge_idx']},"
      f" target_dep={TARGET_DEP})")
print(f"{'='*76}")
print(f"{'T(K)':>5}  {'geo_AR':>8}  {'±':>6}  {'facet_AR':>9}  {'±':>6}  "
      f"{'steps':>8}  {'dep%':>5}  {'wall(s)':>9}")
print('-' * 76)
total_wall = 0.0
for T in TEMPERATURES:
    recs = [r for r in all_records if r['T_K'] == T]
    if not recs:
        continue
    geo_mean  = np.mean([r['geo_AR']   for r in recs])
    geo_std   = np.std( [r['geo_AR']   for r in recs])
    far_mean  = np.mean([r['facet_AR'] for r in recs])
    far_std   = np.std( [r['facet_AR'] for r in recs])
    st_mean   = np.mean([len(r['event_log']) for r in recs])
    dp_mean   = np.mean([r['dep_pct']  for r in recs])
    t_wall    = np.sum( [r['wall_s']   for r in recs])
    total_wall += t_wall
    print(f"{T:>5}  {geo_mean:>8.4f}  {geo_std:>6.4f}  "
          f"{far_mean:>9.3f}  {far_std:>6.3f}  "
          f"{st_mean:>8.0f}  {dp_mean:>4.1f}%  {t_wall:>9.0f}s")
print('-' * 76)
print(f"Seed geo_AR (before growth): {seed_geo:.4f}")
print(f"Total wall time: {total_wall:.0f}s  ({total_wall/3600:.2f} hr)")
print()
print(f"Outputs:")
print(f"  {SUMMARY_FILE}     — all records (positions, CN arrays, event logs)")
print(f"  {CSV_FILE} — tabular summary (geo_AR, facet_AR, timing per run)")
print(f"  final_T*K_s*_silver_500.xyz  — final structure per (T, seed)")
print(f"  sweep_T*K_s*_silver_500.xyz  — trajectory per (T, seed)")
