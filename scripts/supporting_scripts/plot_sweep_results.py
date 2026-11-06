# plot_sweep_results.py
# Reads sweep_results.csv and sweep_results_by_seed.csv (written by
# export_sweep_tables.py) and regenerates:
#   results/fig1_geometric_AR_vs_T.png
#   results/fig2_facet_populations_vs_T.png
#
# Run from the Final/ directory:
#   python scripts/plot_sweep_results.py
#
# Always run export_sweep_tables.py first to ensure the CSVs are up to date.

import numpy as np
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT     = os.path.join(BASE, 'results')
SUMMARY = os.path.join(OUT, 'sweep_results.csv')
BY_SEED = os.path.join(OUT, 'sweep_results_by_seed.csv')

# ── load per-temperature summary ──────────────────────────────────────────────
summary = []
with open(SUMMARY, newline='') as f:
    for row in csv.DictReader(f):
        summary.append({k: float(v) if k != 'T_K' else int(v)
                        for k, v in row.items()})

T_arr      = np.array([r['T_K']           for r in summary])
geo_mean   = np.array([r['geo_AR_mean']   for r in summary])
geo_std    = np.array([r['geo_AR_std']    for r in summary])
n111_mean  = np.array([r['N_111_mean']    for r in summary])
n111_std   = np.array([r['N_111_std']     for r in summary])
n100_mean  = np.array([r['N_100_mean']    for r in summary])
n100_std   = np.array([r['N_100_std']     for r in summary])

# ── load per-seed detail ──────────────────────────────────────────────────────
by_seed = []
with open(BY_SEED, newline='') as f:
    for row in csv.DictReader(f):
        by_seed.append({
            'T_K':    int(row['T_K']),
            'seed':   int(row['seed']),
            'geo_ar': float(row['geo_AR']),
            'n111':   int(row['N_111']),
            'n100':   int(row['N_100']),
        })

# seed reference (truncated octahedron, un-grown)
SEED_GEO_AR = 1.0000
SEED_N111   = 384
SEED_N100   = 54

# ── fig 1 — geometric AR vs T ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))

ax1.axhline(SEED_GEO_AR, color='#888888', lw=1.2, ls='--',
            label=f'Seed (un-grown, AR = {SEED_GEO_AR:.4f})')

np.random.seed(0)
for r in by_seed:
    jitter = np.random.uniform(-0.8, 0.8)
    ax1.scatter(r['T_K'] + jitter, r['geo_ar'],
                color='#4878CF', alpha=0.45, s=28, zorder=3, linewidths=0)

ax1.errorbar(T_arr, geo_mean, yerr=geo_std,
             fmt='o', color='#1a3a6b', ms=7, lw=2, capsize=5, capthick=1.8,
             elinewidth=1.8, zorder=5, label='Mean ± std (3 seeds)')

ax1.set_xlabel('Temperature (K)', fontsize=13)
ax1.set_ylabel('Geometric aspect ratio  $L_1 / \\langle L_2, L_3 \\rangle$', fontsize=13)
ax1.set_title('Au nanorod growth: geometric shape after 100 depositions\n'
              '(PCA on final atom positions, 7 temperatures × 3 seeds)',
              fontsize=11, pad=10)

ax1.set_xlim(273, 347)
y_lo = min(SEED_GEO_AR, (geo_mean - geo_std).min()) - 0.005
y_hi = (geo_mean + geo_std).max() + 0.010
ax1.set_ylim(y_lo, y_hi)

ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, lw=0.4, alpha=0.5)
ax1.tick_params(labelsize=11)

ax1.annotate('Null result: AR ≈ 1.01–1.05\n(indistinguishable from sphere)',
             xy=(310, geo_mean[T_arr == 310][0]),
             xytext=(315, y_lo + 0.006),
             fontsize=9, color='#1a3a6b',
             arrowprops=dict(arrowstyle='->', color='#1a3a6b', lw=1.1))

fig1.tight_layout()
p1 = os.path.join(OUT, 'fig1_geometric_AR_vs_T.png')
fig1.savefig(p1, dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"Figure 1 saved: {p1}")

# ── fig 2 — N(111) and N(100) vs T ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

ax2.axhline(SEED_N111, color='#2ca02c', lw=1.1, ls='--', alpha=0.7,
            label=f'Seed N({{111}}) = {SEED_N111}')
ax2.axhline(SEED_N100, color='#d62728', lw=1.1, ls='--', alpha=0.7,
            label=f'Seed N({{100}}) = {SEED_N100}')

np.random.seed(1)
for r in by_seed:
    jit = np.random.uniform(-0.8, 0.8)
    ax2.scatter(r['T_K'] + jit, r['n111'],
                color='#2ca02c', alpha=0.35, s=25, linewidths=0)
    ax2.scatter(r['T_K'] + jit, r['n100'],
                color='#d62728', alpha=0.35, s=25, linewidths=0)

ax2.errorbar(T_arr, n111_mean, yerr=n111_std,
             fmt='o-', color='#1a6b1a', ms=7, lw=2, capsize=5, capthick=1.8,
             elinewidth=1.8, zorder=5, label='Mean N({111}) ± std')
ax2.errorbar(T_arr, n100_mean, yerr=n100_std,
             fmt='s-', color='#8b0000', ms=7, lw=2, capsize=5, capthick=1.8,
             elinewidth=1.8, zorder=5, label='Mean N({100}) ± std')

ax2.set_xlabel('Temperature (K)', fontsize=13)
ax2.set_ylabel('Surface atom count', fontsize=13)
ax2.set_title('Facet population evolution after 100 depositions\n'
              'N({111}) = CN=9 surface atoms,  N({100}) = CN=8 surface atoms',
              fontsize=11, pad=10)

ax2.set_xlim(273, 347)
ax2.legend(fontsize=10, framealpha=0.9, loc='center right')
ax2.grid(True, lw=0.4, alpha=0.5)
ax2.tick_params(labelsize=11)

ax2.annotate('N({111}) drops: CTAB barely\nblocks {111} (θ=0.05),\nso most depositions land\non {111} and bury terrace atoms',
             xy=(T_arr[3], n111_mean[3]),
             xytext=(295, n111_mean.min() - 25),
             fontsize=8.5, color='#1a6b1a',
             arrowprops=dict(arrowstyle='->', color='#1a6b1a', lw=1.0))

ax2.annotate('N({100}) grows\nwith T (more diffusion\nexposes {100}-type sites)',
             xy=(T_arr[-1], n100_mean[-1]),
             xytext=(303, n100_mean.max() + 18),
             fontsize=8.5, color='#8b0000',
             arrowprops=dict(arrowstyle='->', color='#8b0000', lw=1.0))

fig2.tight_layout()
p2 = os.path.join(OUT, 'fig2_facet_populations_vs_T.png')
fig2.savefig(p2, dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"Figure 2 saved: {p2}")
