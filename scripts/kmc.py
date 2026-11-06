import numpy as np
from ase import Atoms as AseAtoms
from ase.io import write as ase_write

from coordination import (compute_all_cn, update_cn_local,
                           surface_atom_mask, assert_no_overlaps)
from diffusion import diffusion_rate
from deposition import deposition_rate, K_DEP_BASE
from neighbors import candidate_hop_sites, surface_adsorption_sites


def _build_event_catalog(positions, cn, T, k_dep, is_silver_proximal):
    """
    Build the complete event catalog for the current configuration.

    Rebuilds from scratch each step — acceptable for the validation run
    (<=500 steps). For production, replace with an incremental update that
    only re-evaluates events for atoms whose CN changed after the last event.

    Returns
    -------
    rates      : (E,) float array of event rates (s^-1)
    event_types: list of E strings, 'diff' or 'dep'
    event_data : list of E items:
                   'diff' -> (atom_idx: int, target_pos: (3,) array)
                   'dep'  -> target_pos: (3,) array
    """
    rates       = []
    event_types = []
    event_data  = []

    # Diffusion events: one entry per (surface atom, valid hop site) pair
    for idx in np.where(surface_atom_mask(cn))[0]:
        hop_sites = candidate_hop_sites(idx, positions)
        if len(hop_sites) == 0:
            continue
        r = diffusion_rate(float(cn[idx]), T)
        for site in hop_sites:
            rates.append(r)
            event_types.append('diff')
            event_data.append((int(idx), site.copy()))

    # Deposition events: one entry per empty surface site
    dep_sites, parent_cns, site_is_ag, site_cns_raw = surface_adsorption_sites(
        positions, cn, is_silver_proximal
    )
    for site, pcn, is_ag, scn in zip(dep_sites, parent_cns, site_is_ag, site_cns_raw):
        rates.append(deposition_rate(int(pcn), k_dep, bool(is_ag), int(scn)))
        event_types.append('dep')
        event_data.append(site.copy())

    return np.array(rates, dtype=float), event_types, event_data


def _checkpoint(positions, path, step, t):
    atoms = AseAtoms('Au' * len(positions), positions=positions)
    atoms.info['step']       = step
    atoms.info['kmc_time_s'] = t
    ase_write(path, atoms, format='extxyz', append=True)
    print(f"  [step {step:>6d}]  t = {t:.3e} s   N = {len(positions)} atoms  -> {path}")


def run_kmc(
    initial_positions,
    T,
    n_steps,
    k_dep              = K_DEP_BASE,
    rng                = None,
    debug              = False,
    checkpoint_every   = 100,
    checkpoint_path    = 'kmc_checkpoint.xyz',
    target_dep_events  = None,
    silver_rng_seed    = None,
):
    """
    Gillespie KMC loop for Au nanorod growth.

    Two event types only: surface diffusion and deposition.
    Detachment is excluded — CTAB-directed growth is kinetically trapped.

    Parameters
    ----------
    initial_positions : (N, 3) float array — seed atom positions, A
    T                 : float — temperature, K
    n_steps           : int   — maximum KMC steps (always respected as a safety cap)
    k_dep             : float — base deposition rate constant, s^-1 (tunable)
    rng               : np.random.Generator; seeded with 42 if None
    debug             : bool  — run assert_no_overlaps + CN=0 check after every
                        event (slow; use for the short validation run only)
    checkpoint_every  : int   — write xyz snapshot every N completed steps
    checkpoint_path   : str   — trajectory file for checkpoints (extxyz, appended)
    target_dep_events : int or None — if set, stop as soon as this many deposition
                        events have fired (whichever comes first: target or n_steps).
                        Use this instead of a fixed n_steps for the temperature sweep
                        so every run incorporates exactly the same number of atoms.
    silver_rng_seed   : int or None — if set, call build_silver_mask() once on the
                        initial geometry and apply the silver blocking factor throughout.
                        None = Phase 1 mode (no silver, rates unchanged).

    Returns
    -------
    dict with keys:
      'positions' : (N_final, 3) final atom positions
      'cn'        : (N_final,)   final CN array
      'time'      : float        final KMC time, s
      'n_diff'    : int          diffusion events executed
      'n_dep'     : int          deposition events executed
      'times'     : (steps,)     KMC time after each completed step
      'n_atoms'   : (steps,)     atom count after each completed step
      'event_log' : list of str  'diff' or 'dep' per completed step
    """
    if rng is None:
        rng = np.random.default_rng(42)

    positions = initial_positions.copy().astype(float)
    cn        = compute_all_cn(positions)

    # Silver mask — computed once on initial geometry, held fixed throughout.
    # silver_rng_seed=None → all-False mask (Phase 1 / no-silver mode).
    if silver_rng_seed is not None:
        from silver import build_silver_mask
        is_silver_proximal, _ag_info = build_silver_mask(
            positions, cn, rng_seed=silver_rng_seed
        )
        print(f"  Silver mask: {_ag_info['n_proximal']} proximal atoms "
              f"(ridge {_ag_info['selected_ridge_idx']}, "
              f"rng_seed={silver_rng_seed})")
    else:
        is_silver_proximal = None

    t         = 0.0
    n_diff    = 0
    n_dep     = 0
    completed = 0

    times      = np.zeros(n_steps)
    n_atoms    = np.zeros(n_steps, dtype=int)
    event_log  = []          # 'diff' or 'dep' per completed step

    for step in range(n_steps):

        # Build event catalog
        rates, event_types, event_data = _build_event_catalog(
            positions, cn, T, k_dep, is_silver_proximal
        )

        if rates.size == 0:
            print(f"Step {step}: no valid events — simulation stalled.")
            break

        R_total = rates.sum()

        # Gillespie time advance: dt ~ Exp(1/R_total)
        dt  = -np.log(rng.random()) / R_total
        t  += dt

        # Event selection: first event whose cumulative rate exceeds u * R_total
        u       = rng.random()
        event_i = int(np.searchsorted(np.cumsum(rates), u * R_total))
        event_i = min(event_i, len(rates) - 1)   # guard against fp rounding

        etype = event_types[event_i]
        edata = event_data[event_i]

        # Execute event
        if etype == 'diff':
            atom_idx, target_pos = edata
            pos_old = positions[atom_idx].copy()
            positions[atom_idx] = target_pos
            update_cn_local(positions, cn,
                            np.vstack([pos_old, target_pos]))
            n_diff += 1

        else:  # 'dep'
            new_pos   = edata
            positions = np.vstack([positions, new_pos])
            cn        = np.append(cn, 0)          # placeholder; updated below
            update_cn_local(positions, cn,
                            new_pos[np.newaxis, :])
            n_dep += 1
            # New atoms don't inherit silver-proximal status — blocking already
            # acted on the parent atom's deposition rate.
            if is_silver_proximal is not None:
                is_silver_proximal = np.append(is_silver_proximal, False)

        # Correctness guards — debug mode only; disable for production runs
        if debug:
            assert_no_overlaps(positions)
            isolated = np.where(cn == 0)[0]
            assert len(isolated) == 0, (
                f"Step {step}: {len(isolated)} isolated atom(s) with CN=0 "
                f"at indices {isolated.tolist()}. "
                "Check connectivity filter in candidate_hop_sites / "
                "surface_adsorption_sites."
            )

        # Record
        times[step]   = t
        n_atoms[step] = len(positions)
        event_log.append(etype)
        completed     = step + 1

        # Checkpoint
        if completed % checkpoint_every == 0:
            _checkpoint(positions, checkpoint_path, completed, t)

        # Deposition-count stopping criterion (checked after recording so the
        # triggering event is always included in the log before we exit)
        if target_dep_events is not None and n_dep >= target_dep_events:
            print(f"  target_dep_events={target_dep_events} reached at step {completed}.")
            break

    _checkpoint(positions,
                checkpoint_path.replace('.xyz', '_final.xyz'),
                completed, t)

    return {
        'positions': positions,
        'cn':        cn,
        'time':      t,
        'n_diff':    n_diff,
        'n_dep':     n_dep,
        'times':     times[:completed],
        'n_atoms':   n_atoms[:completed],
        'event_log': event_log,
    }
