"""
silver.py — phenomenological silver symmetry-breaking for Phase 2 KMC

Physical basis
--------------
Silver ions (via underpotential deposition, UPD) stabilize whichever
{111}–{111} edge truncation appears first stochastically, locking it in
permanently. CTAB + Br⁻ then forms a tight blocking layer on the
silver-stabilized surface, suppressing gold deposition there. Unpassivated
{111} faces remain open → preferential gold deposition → elongation.

The symmetry-breaking is inherently stochastic (random ridge selection) but
fully reproducible for a given rng_seed.

Literature
----------
Walsh et al., ACS Nano 9, 715 (2015)  — stochastic truncation, 4–6 nm threshold
Walsh et al., Acc. Chem. Res. (2017)  — Ag stabilises the new surface that breaks symmetry
Personick et al., Nano Lett. 11, 3394 (2011) — Ag UPD (verify the {110} / CTAB-Br claim)
"""
from __future__ import annotations
import numpy as np
from collections import deque

CN_CUTOFF = 3.0   # Å — must match coordination.py throughout


# ── Ridge detection ───────────────────────────────────────────────────────────

def find_edge_atoms(cn: np.ndarray) -> np.ndarray:
    """
    Return indices of all CN=7 atoms.

    CN=7 atoms sit on two distinct edge types in the truncated octahedron:
      {111}–{111} ridges : the 12 edges of the original octahedron
      {111}–{100} edges  : the 24 edges bordering the truncation faces

    Both have CN=7 after BFGS relaxation.  The two types are discriminated
    by cluster size in filter_111_ridges().
    """
    return np.where(cn == 7)[0]


def cluster_ridges(
    positions: np.ndarray,
    edge_indices: np.ndarray,
    cutoff: float = CN_CUTOFF,
) -> list[np.ndarray]:
    """
    BFS connectivity clustering of CN=7 atoms.

    Two CN=7 atoms belong to the same cluster when they are within `cutoff`
    of each other.  At 3.0 Å this correctly separates all CN=7 atoms into
    36 clusters: 12 {111}–{111} clusters (size 4) and 24 {111}–{100}
    clusters (size 3) — confirmed on the relaxed 4 nm seed.

    Returns a list of atom-index arrays, one per cluster.
    """
    cutoff2  = cutoff ** 2
    edge_pos = positions[edge_indices]
    n        = len(edge_indices)
    visited  = np.zeros(n, dtype=bool)
    ridges   = []

    for start in range(n):
        if visited[start]:
            continue
        cluster: list[int] = []
        queue = deque([start])
        visited[start] = True
        while queue:
            i = queue.popleft()
            cluster.append(i)
            diff  = edge_pos - edge_pos[i]
            dist2 = np.einsum('ij,ij->i', diff, diff)
            for nb in np.where((dist2 > 0) & (dist2 <= cutoff2))[0]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        ridges.append(edge_indices[np.array(cluster)])

    return ridges


def filter_111_ridges(all_clusters: list[np.ndarray]) -> list[np.ndarray]:
    """
    Retain only the {111}–{111} ridge clusters, discarding {111}–{100} edges.

    In the 4 nm truncated-octahedron seed, the two edge types separate
    cleanly by cluster size after BFS at 3.0 Å:
      {111}–{111} ridges : 12 clusters, each 4 CN=7 atoms  ← silver target
      {111}–{100} edges  : 24 clusters, each 3 CN=7 atoms  ← not silver target

    Silver targets the {111}–{111} boundary because that is where thermal
    fluctuations create higher-index truncations stabilised by UPD (Walsh
    et al. 2015; Etheridge et al. 2017).  The {111}–{100} edges are already
    {100}-adjacent and do not require silver to be blocked.

    Strategy: retain only clusters of the maximum observed cluster size.
    This is robust — {111}–{111} ridges are always longer than {111}–{100}
    edges for this particle geometry (length=14, cutoff=4).
    """
    if not all_clusters:
        return []
    max_size = max(len(c) for c in all_clusters)
    ridges   = [c for c in all_clusters if len(c) == max_size]
    if len(ridges) != 12:
        import warnings
        warnings.warn(
            f"Expected 12 {{111}}–{{111}} ridges, found {len(ridges)} "
            f"(max cluster size={max_size}).  "
            "Verify seed geometry if this is unexpected.",
            stacklevel=2,
        )
    return ridges


# Ridge selection

def select_random_ridge(
    ridges: list[np.ndarray],
    rng_seed: int,
) -> tuple[int, np.ndarray]:
    """
    Pick one ridge uniformly at random.

    The randomness models which stochastic truncation the silver happens to
    stabilise. rng_seed makes the choice reproducible across runs; each
    seed value samples a different symmetry-breaking axis.

    Returns (ridge_index, atom_index_array).
    """
    rng = np.random.default_rng(rng_seed)
    idx = int(rng.integers(0, len(ridges)))
    return idx, ridges[idx]


# Corner detection

def find_endpoint_corners(
    positions: np.ndarray,
    cn: np.ndarray,
    ridge_indices: np.ndarray,
    cutoff: float = CN_CUTOFF,
) -> np.ndarray:
    """
    Find CN=6 corner atoms at the two ends of the selected ridge.

    Corners are octahedron vertices where the ridge terminates and adjoins
    two other ridges.  Any CN=6 atom within `cutoff` of a ridge atom is an
    endpoint corner.  Expected: exactly 2 per ridge.
    """
    cutoff2          = cutoff ** 2
    corner_candidates = np.where(cn == 6)[0]
    ridge_pos        = positions[ridge_indices]   # (R, 3)
    corners: list[int] = []

    for ci in corner_candidates:
        diff  = ridge_pos - positions[ci]         # (R, 3)
        dist2 = np.einsum('ij,ij->i', diff, diff) # (R,)
        if np.any((dist2 > 0) & (dist2 <= cutoff2)):
            corners.append(int(ci))

    return np.array(corners, dtype=np.intp)


# Main pipeline

def build_silver_mask(
    positions: np.ndarray,
    cn: np.ndarray,
    rng_seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Detect ridges → select one → flag silver-proximal atoms.

    Parameters
    ----------
    positions : (N, 3) float, Å  — atom positions (seed or grown particle)
    cn        : (N,)  int        — precomputed coordination numbers
    rng_seed  : int              — controls which ridge is stabilised

    Returns
    -------
    is_silver_proximal : (N,) bool
        True for every atom that is silver-stabilised (ridge + corners) or
        within 3.0 Å of a stabilised atom.  Passed into deposition.py as
        the θ_Ag_local selector:
            R_dep = k_dep × (1 − θ_CTAB) × (1 − θ_Ag × is_silver_proximal)

    info : dict
        Diagnostic keys: ridge_count, ridge_sizes, selected_ridge_idx,
        ridge_atom_indices, corner_atom_indices, n_stabilized, n_proximal.
    """
    N       = len(positions)
    cutoff2 = CN_CUTOFF ** 2

    # Steps 1–2: find, cluster, then filter to {111}–{111} ridges only
    edge_indices = find_edge_atoms(cn)
    all_clusters = cluster_ridges(positions, edge_indices)
    ridges       = filter_111_ridges(all_clusters)

    # Step 3: randomly select one ridge
    ridge_idx, ridge_atoms = select_random_ridge(ridges, rng_seed)

    # Step 4: CN=6 corner atoms at the ends of the selected ridge
    corner_atoms = find_endpoint_corners(positions, cn, ridge_atoms)

    # Step 5: silver-stabilised = ridge + corners
    stabilized = np.union1d(ridge_atoms, corner_atoms)

    # Step 6: silver-proximal = stabilised ∪ {atoms within 3.0 Å of any stabilised atom}
    is_silver_proximal = np.zeros(N, dtype=bool)
    is_silver_proximal[stabilized] = True

    stab_pos = positions[stabilized]                              # (S, 3)
    diffs    = positions[:, np.newaxis, :] - stab_pos[np.newaxis, :, :]  # (N, S, 3)
    dist2    = np.einsum('ijk,ijk->ij', diffs, diffs)            # (N, S)
    is_silver_proximal |= np.any(dist2 <= cutoff2, axis=1)

    return is_silver_proximal, {
        'total_cn7_clusters':  len(all_clusters),
        'ridge_count':         len(ridges),
        'ridge_sizes':         [len(r) for r in ridges],
        'selected_ridge_idx':  ridge_idx,
        'ridge_atom_indices':  ridge_atoms,
        'corner_atom_indices': corner_atoms,
        'n_stabilized':        len(stabilized),
        'n_proximal':          int(is_silver_proximal.sum()),
    }


# Diagnostics

def print_silver_report(cn: np.ndarray, info: dict) -> None:
    """Print a human-readable summary of the silver mask to stdout."""
    n_ridges = info['ridge_count']
    sizes    = sorted(info['ridge_sizes'])

    print("=" * 52)
    print("Silver mask report")
    print("=" * 52)
    total_clusters = info.get('total_cn7_clusters', '?')
    print(f"Total CN=7 clusters   : {total_clusters}  "
          f"(12 {{111}}–{{111}} ridges + 24 {{111}}–{{100}} edges expected)")
    print(f"{{111}}–{{111}} ridges   : {n_ridges}  (expected 12)")
    print(f"Ridge sizes (CN=7)    : {sizes}")
    if n_ridges != 12:
        print(f"  *** WARNING: expected 12 for truncated octahedron ***")

    print(f"\nSelected ridge index  : {info['selected_ridge_idx']}")
    ridge_atoms  = info['ridge_atom_indices']
    corner_atoms = info['corner_atom_indices']

    print(f"Ridge atoms  (CN=7)   : {len(ridge_atoms)} atoms")
    for idx in sorted(ridge_atoms):
        print(f"  [{idx:5d}]  CN={cn[idx]}")

    print(f"\nCorner atoms (CN=6)   : {len(corner_atoms)} atoms  (expected 2)")
    for idx in sorted(corner_atoms):
        print(f"  [{idx:5d}]  CN={cn[idx]}")

    n_stab = info['n_stabilized']
    n_prox = info['n_proximal']
    print(f"\nSilver-stabilised     : {n_stab}"
          f"  (ridge {len(ridge_atoms)} + corners {len(corner_atoms)})")
    print(f"Silver-proximal total : {n_prox}"
          f"  (stabilised + 3.0 Å neighbours)")
    print("=" * 52)


def save_silver_xyz(
    atoms_ase,
    is_silver_proximal: np.ndarray,
    info: dict,
    filename: str = 'octahedron_4nm_silver.xyz',
) -> None:
    """
    Write an extended XYZ file with two per-atom integer columns for OVITO.

      silver_stabilized : 1 for the core ridge + corner atoms (direct UPD site)
      silver_proximal   : 1 for stabilised atoms + their 3.0 Å neighbours

    In OVITO:
      Particles → Color Coding → Property = silver_proximal
        (0 = unaffected Au, 1 = CTAB+Ag blocked)
      Or: Particles → Filter → silver_stabilized == 1
        to isolate just the core ridge and inspect its geometry.
    """
    from ase.io import write

    out = atoms_ase.copy()
    N   = len(out)

    stabilized   = np.union1d(info['ridge_atom_indices'], info['corner_atom_indices'])
    is_stab      = np.zeros(N, dtype=np.int32)
    is_stab[stabilized] = 1

    out.arrays['silver_stabilized'] = is_stab
    out.arrays['silver_proximal']   = is_silver_proximal.astype(np.int32)

    write(filename, out, format='extxyz')
    print(f"Saved → {filename}")
    print("  OVITO: Color Coding → silver_proximal  (0=normal, 1=blocked)")
    print("         Filter → silver_stabilized==1 to isolate the core ridge")


# CLI entry point

if __name__ == '__main__':
    import sys
    from ase.io import read
    from coordination import compute_all_cn

    seed_file = 'octahedron_4nm_relaxed.xyz'
    rng_seed  = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    print(f"Loading {seed_file}  (rng_seed={rng_seed})")
    atoms = read(seed_file)
    pos   = atoms.get_positions()
    cn    = compute_all_cn(pos)

    mask, info = build_silver_mask(pos, cn, rng_seed=rng_seed)
    print_silver_report(cn, info)
    save_silver_xyz(atoms, mask, info, 'octahedron_4nm_silver.xyz')
