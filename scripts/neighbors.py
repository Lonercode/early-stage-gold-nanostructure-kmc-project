import numpy as np
from coordination import CN_CUTOFF, CN_BULK

AU_LATTICE = 4.078          # Å, Au FCC lattice parameter (bulk, ~300 K)
AU_NN_DIST = AU_LATTICE / np.sqrt(2)   # ≈ 2.884 Å — FCC nearest-neighbour distance
MIN_SEPARATION = 2.0        # Å — floor below which two atoms cannot coexist

# 12 FCC nearest-neighbour displacement vectors: (a/2)(±1,±1,0) and permutations.
# Magnitude: (a/2)√2 = a/√2 ≈ 2.884 Å.
_FCC_NN_VECS = (AU_LATTICE / 2.0) * np.array([
    [ 1,  1,  0], [ 1, -1,  0], [-1,  1,  0], [-1, -1,  0],
    [ 1,  0,  1], [ 1,  0, -1], [-1,  0,  1], [-1,  0, -1],
    [ 0,  1,  1], [ 0,  1, -1], [ 0, -1,  1], [ 0, -1, -1],
], dtype=float)


def candidate_hop_sites(
    atom_idx: int,
    positions: np.ndarray,
    cutoff: float = CN_CUTOFF,
    min_sep: float = MIN_SEPARATION,
) -> np.ndarray:
    """
    Find all valid FCC sites that atom `atom_idx` could hop to in a diffusion event.

    Generates 12 candidate positions (current pos + each FCC NN vector), then
    retains only those satisfying:
      1. Empty     — no OTHER atom within min_sep (2.0 Å)
      2. Connected — at least one OTHER atom within cutoff (3.0 Å),
                     so the hopping atom doesn't become a disconnected stray

    The hopping atom itself is excluded from both checks — it is vacating its site.

    Parameters
    ----------
    atom_idx  : index of the diffusing atom
    positions : (N, 3) current atom positions, Å
    cutoff    : neighbour-shell radius for connectivity check, Å
    min_sep   : minimum allowed distance to any existing atom, Å

    Returns
    -------
    np.ndarray of shape (M, 3) — valid target positions; shape (0, 3) if trapped.
    """
    candidates = positions[atom_idx] + _FCC_NN_VECS   # (12, 3)

    # Exclude the hopping atom — it is leaving its current site
    mask   = np.ones(len(positions), dtype=bool)
    mask[atom_idx] = False
    others = positions[mask]                           # (N-1, 3)

    diff  = candidates[:, np.newaxis, :] - others[np.newaxis, :, :]  # (12, N-1, 3)
    dist2 = np.einsum('coj,coj->co', diff, diff)                      # (12, N-1)

    empty     = dist2.min(axis=1)             >= min_sep ** 2
    connected = (dist2 <= cutoff ** 2).sum(axis=1) >= 1

    return candidates[empty & connected]


def surface_adsorption_sites(
    positions: np.ndarray,
    cn: np.ndarray,
    is_silver_proximal: np.ndarray | None = None,
    cutoff: float = CN_CUTOFF,
    min_sep: float = MIN_SEPARATION,
) -> tuple:
    """
    Find all empty FCC sites adjacent to the current surface — candidate
    landing positions for deposition events.

    Returns
    -------
    sites               : (M, 3) float — candidate deposition positions, Å
    parent_cns          : (M,)   int32 — CN of the surface atom that generated
                          each site (first-encounter on dedup); used for CTAB
                          facet classification in deposition_rate().
    site_is_silver_prox : (M,)   bool  — True when the parent atom is in the
                          silver-proximal mask.
    site_cn_raws        : (M,)   int32 — raw connectivity count of the candidate
                          site itself: number of existing atoms within CN_CUTOFF
                          of that position. Used for BCF edge/kink premium in
                          deposition_rate(). Always >= 3 (connectivity floor).

    CN convention note
    ------------------
    parent_cns and site_cn_raws are on DIFFERENT scales and serve different
    purposes:
      parent_cns   — CN of the surface atom below; range ~6–12; used by
                     facet_from_cn() to classify {111}/{100}/{110}.
      site_cn_raws — CN of the empty candidate site; range 3–7+; used by
                     the BCF premium (more bonds formed on landing = stickier).
    Do NOT pass site_cn_raws to facet_from_cn() — it would misclassify every
    site as {110}.

    Deduplication
    -------------
    Multiple surface atoms can point at the same hollow/bridge site.  Sites
    within 0.5 Å of each other are treated as identical; the first-encounter
    parent CN and site_cn_raw are kept.  site_cn_raw is a property of the
    candidate position (distance to all atoms), so both parents compute the
    same value — first-encounter is consistent.
    """
    surface_indices = np.where((cn > 0) & (cn <= CN_BULK))[0]
    cutoff2  = cutoff  ** 2
    min_sep2 = min_sep ** 2

    seen: dict = {}   # quantised grid key → (position, parent_cn, is_ag_prox, site_cn_raw)

    for idx in surface_indices:
        candidates = positions[idx] + _FCC_NN_VECS         # (12, 3)

        diff  = candidates[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (12, N, 3)
        dist2 = np.einsum('coj,coj->co', diff, diff)                         # (12, N)

        cn_sites  = (dist2 <= cutoff2).sum(axis=1)           # (12,) — site connectivity
        empty     = dist2.min(axis=1)    >= min_sep2
        connected = cn_sites             >= 3    # ≥3 neighbours: minimum for a physical
                                                 # adsorption site ({111} hollow has exactly 3)

        parent_cn    = int(cn[idx])
        parent_is_ag = bool(is_silver_proximal[idx]) \
                       if is_silver_proximal is not None else False

        for site, ok, scn in zip(candidates, empty & connected, cn_sites):
            if not ok:
                continue
            key = tuple(np.round(site / 0.5).astype(int))
            if key not in seen:
                seen[key] = (site, parent_cn, parent_is_ag, int(scn))

    if not seen:
        return (np.empty((0, 3)),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=bool),
                np.empty(0, dtype=np.int32))

    sites_list, cns_list, ag_list, scn_list = zip(*seen.values())
    return (np.array(sites_list),
            np.array(cns_list, dtype=np.int32),
            np.array(ag_list,  dtype=bool),
            np.array(scn_list, dtype=np.int32))
