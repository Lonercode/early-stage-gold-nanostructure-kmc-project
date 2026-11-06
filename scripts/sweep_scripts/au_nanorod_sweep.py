# au_nanorod_sweep.py — Kaggle production temperature sweep
#
# Upload to Kaggle alongside:
#   octahedron_4nm_relaxed.xyz
#   coordination.py  diffusion.py  deposition.py  neighbors.py  kmc.py
#
# Each (temperature, seed) pair runs until TARGET_DEP deposition events fire,
# guaranteeing every run incorporates exactly the same number of atoms.
# sweep_summary.npy is written after every completed run so a session crash
# never loses more than one run's worth of data.

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ase', '-q'])

import os, time
import numpy as np
from ase.io import read, write as ase_write
from ase import Atoms as AseAtoms

from coordination import compute_all_cn
from kmc import run_kmc

# ── Sweep parameters ──────────────────────────────────────────────────────────
SEED_FILE        = 'octahedron_4nm_relaxed.xyz'
K_DEP            = 3.5e7    # s^-1; validated at T=300 K, held fixed across sweep
                             # (represents fixed solution-side Au flux; see methods)
TARGET_DEP       = 100      # deposition events per run; guarantees equal atom count
N_STEPS_CAP      = 500_000  # safety cap — should never be reached
CHECKPOINT_EVERY = 500      # write xyz frame every N steps

TEMPERATURES = [280, 295, 300, 310, 320, 330, 340]  # K — 360K excluded: outside
                                                    # Park et al. comparison range
                                                    # (276–315K) and disproportionately
                                                    # expensive (~5hr/seed at target=100)
RNG_SEEDS    = [42, 7, 123]                               # 3 independent runs per T

SUMMARY_FILE = 'sweep_summary.npy'


# ── Aspect ratio from surface facet atom counts ───────────────────────────────
def compute_aspect_ratio(cn: np.ndarray) -> float:
    """
    N({100}) / N({111}) surface-atom ratio as a proxy for nanorod aspect ratio.

    As the rod elongates at roughly constant diameter:
      N({111}) ~ 2 end caps ~ d^2  — fixed by diameter, barely grows with length
      N({100}) ~ side walls  ~ d*L — grows with length

    So N({100}) / N({111}) ~ L/d, i.e. it increases monotonically with aspect
    ratio.  The inverse (N({111})/N({100})) would decrease with elongation.

    CN=9 surface atoms → {111}-like (end facets)
    CN=8 surface atoms → {100}-like (side facets)

    Returns nan if no CN=9 atoms present (degenerate geometry).
    """
    n_111 = int((cn == 9).sum())
    n_100 = int((cn == 8).sum())
    return n_100 / n_111 if n_111 > 0 else float('nan')


# ── Incremental summary save ──────────────────────────────────────────────────
def save_summary(records: list) -> None:
    np.save(SUMMARY_FILE, records)


# ── Load seed ─────────────────────────────────────────────────────────────────
seed_atoms = read(SEED_FILE)
pos0       = seed_atoms.get_positions().astype(float)
seed_cn    = compute_all_cn(pos0)
seed_ar    = compute_aspect_ratio(seed_cn)

print(f"Seed: {len(pos0)} atoms  |  seed AR = {seed_ar:.3f}  |  {SEED_FILE}")
print(f"Temperatures : {TEMPERATURES} K")
print(f"Seeds        : {RNG_SEEDS}")
print(f"Target dep   : {TARGET_DEP} events per run")
print(f"k_dep        : {K_DEP:.2e} s^-1\n")

# ── Resume support: load existing summary if present ─────────────────────────
if os.path.exists(SUMMARY_FILE):
    existing = list(np.load(SUMMARY_FILE, allow_pickle=True))
    done = {(r['T_K'], r['rng_seed']) for r in existing}
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

        label     = f'T{int(T)}K_s{rng_seed}'
        ckpt_path = f'sweep_{label}.xyz'
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        print(f"  T={T} K  seed={rng_seed}")
        t_wall = time.perf_counter()

        result = run_kmc(
            pos0,
            T                = T,
            n_steps          = N_STEPS_CAP,
            k_dep            = K_DEP,
            rng              = np.random.default_rng(rng_seed),
            debug            = False,
            checkpoint_every = CHECKPOINT_EVERY,
            checkpoint_path  = ckpt_path,
            target_dep_events= TARGET_DEP,
        )

        elapsed  = time.perf_counter() - t_wall
        n_steps  = len(result['event_log'])
        n_dep    = result['n_dep']
        n_diff   = result['n_diff']
        final_N  = len(result['positions'])
        kmc_time = result['time']
        dep_pct  = n_dep / n_steps * 100
        ar       = compute_aspect_ratio(result['cn'])

        print(f"    steps={n_steps:,}  dep={n_dep}({dep_pct:.1f}%)  "
              f"diff={n_diff:,}  N={final_N}  AR={ar:.3f}  "
              f"wall={elapsed:.0f}s  ({elapsed/n_steps:.3f} s/step)")

        # Save final structure for this (T, seed)
        final_atoms = AseAtoms('Au' * final_N, positions=result['positions'])
        final_atoms.info.update({
            'T_K': T, 'rng_seed': rng_seed,
            'kmc_time_s': kmc_time, 'n_dep': n_dep,
            'n_diff': n_diff, 'k_dep': K_DEP, 'aspect_ratio': ar,
        })
        ase_write(f'final_{label}.xyz', final_atoms, format='extxyz')

        record = {
            'T_K':      T,
            'rng_seed': rng_seed,
            'n_steps':  n_steps,
            'n_dep':    n_dep,
            'n_diff':   n_diff,
            'dep_pct':  dep_pct,
            'final_N':  final_N,
            'kmc_time': kmc_time,
            'wall_s':   elapsed,
            'ar':       ar,
            'positions': result['positions'].copy(),
            'cn':        result['cn'].copy(),
            'event_log': result['event_log'],
        }
        all_records.append(record)
        T_records.append(record)

        # Incremental save after every run — crash-safe
        save_summary(all_records)
        print(f"    -> {SUMMARY_FILE} updated ({len(all_records)} runs total)\n")

    # Per-temperature summary across seeds
    ars   = [r['ar']      for r in T_records]
    walls = [r['wall_s']  for r in T_records]
    ns    = [r['n_steps'] for r in T_records]
    print(f"  T={T}K summary: AR={np.mean(ars):.3f}±{np.std(ars):.3f}  "
          f"steps={np.mean(ns):.0f}±{np.std(ns):.0f}  "
          f"wall={np.sum(walls):.0f}s total\n")

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"SWEEP COMPLETE — {len(all_records)} runs")
print(f"{'='*72}")
print(f"{'T(K)':>5}  {'AR mean':>8}  {'AR std':>7}  {'steps mean':>11}  "
      f"{'dep%':>5}  {'wall/T(s)':>10}")
print('-' * 72)
total_wall = 0.0
for T in TEMPERATURES:
    recs = [r for r in all_records if r['T_K'] == T]
    if not recs:
        continue
    ar_mean  = np.mean([r['ar']      for r in recs])
    ar_std   = np.std( [r['ar']      for r in recs])
    st_mean  = np.mean([r['n_steps'] for r in recs])
    dp_mean  = np.mean([r['dep_pct'] for r in recs])
    t_wall   = np.sum( [r['wall_s']  for r in recs])
    total_wall += t_wall
    print(f"{T:>5}  {ar_mean:>8.3f}  {ar_std:>7.3f}  {st_mean:>11.0f}  "
          f"{dp_mean:>4.1f}%  {t_wall:>10.0f}s")
print('-' * 72)
print(f"Seed AR (before growth): {seed_ar:.3f}")
print(f"Total wall time: {total_wall:.0f}s  ({total_wall/3600:.2f} hr)")
print()
print(f"Outputs:")
print(f"  {SUMMARY_FILE}     — all records (aspect ratios, CN arrays, event logs)")
print(f"  final_T*K_s*.xyz  — final structure per (T, seed)")
print(f"  sweep_T*K_s*.xyz  — trajectory per (T, seed)")
