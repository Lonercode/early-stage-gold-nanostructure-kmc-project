import numpy as np

CN_CUTOFF = 3.0
CN_BULK   = 9     # atoms with CN > CN_BULK are frozen during KMC


def compute_all_cn(positions: np.ndarray, cutoff: float = CN_CUTOFF) -> np.ndarray:
    """
    Compute coordination number for every atom.

    Used at initialization and after any bulk restructuring.
    During the KMC loop, update_cn_local is used.

    Parameters
    ----------
    positions : (N, 3) float array, Å
    cutoff    : neighbour shell radius in Å

    Returns
    -------
    np.ndarray of shape (N,)
    """
    diff  = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]   # (N, N, 3)
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)                         # (N, N)
    in_shell = (dist2 > 0) & (dist2 <= cutoff ** 2)                      # exclude self
    return in_shell.sum(axis=1).astype(np.int32)


def update_cn_local(positions: np.ndarray,
                    cn: np.ndarray,
                    changed_positions: np.ndarray,
                    cutoff: float = CN_CUTOFF) -> None:
    """
    In-place CN refresh for every atom within `cutoff` of any position in
    `changed_positions`.

    Called after every KMC event, passing both the old and new coordinates
    of the atom that moved - that way every neighbour of either site gets a
    fresh count.

    Parameters
    ----------
    positions         : (N, 3) current atom positions (already updated)
    cn                : (N,)   CN array, modified in-place
    changed_positions : (M, 3) old and/or new positions of the event atom(s)
    cutoff            : neighbour shell radius in Å
    """
    cutoff2: float = cutoff ** 2
    to_refresh: set = set()

    for ref in changed_positions:
        diff  = positions - ref
        dist2 = np.einsum('ij,ij->i', diff, diff)
        to_refresh.update(np.where(dist2 <= cutoff2)[0].tolist())

    for j in to_refresh:
        diff_j  = positions - positions[j]
        dist2_j = np.einsum('ij,ij->i', diff_j, diff_j)
        cn[j]   = int(((dist2_j > 0) & (dist2_j <= cutoff2)).sum())


def surface_atom_mask(cn: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True for atoms eligible to participate in KMC events.

    Frozen  : CN > CN_BULK  (bulk-like; excluded per design)
    Isolated: CN == 0       (disconnected stray; should never exist — caught here
                             as a sign of the diffusion-overwrite bug)
    """
    return (cn > 0) & (cn <= CN_BULK)


# Correctness guard - this is run after every event during validation

def assert_no_overlaps(positions: np.ndarray, min_dist: float = 2.0) -> None:
    """
    Assert that no two atoms are closer than min_dist Å.

    Directly catches the diffusion-overwrite bug: an atom placed onto an
    occupied site would produce a near-zero separation.  Run after
    every event during the short validation run.

    Raises AssertionError with the offending pair's distance if triggered.
    """
    diff  = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, np.inf)
    min_sep = np.sqrt(dist2.min())
    assert min_sep >= min_dist, (
        f"Atom overlap: minimum separation {min_sep:.4f} Å < {min_dist} Å — "
        "likely a diffusion-overwrite bug (atom placed on an occupied site)."
    )
