import numpy as np

# CTAB blocking factors
# θ = fraction of surface sites blocked by CTAB on each facet type.
# {111} end facets: low CTAB affinity, low blocking with fast Au deposition
# {100}/{110} side facets: high CTAB affinity, high blocking with slow Au deposition
# These are tunable parameters; set physically motivated defaults here.
THETA_111 = 0.05   # 5% blocked   — {111}, weakly bound CTAB
THETA_100 = 0.80   # 80% blocked  — {100}, strongly bound CTAB
THETA_110 = 0.90   # 90% blocked  — {110}, most strongly bound CTAB
THETA_AG  = 0.90   # Silver UPD blocking — tight CTAB+Br⁻ layer on Ag-stabilised
                   # {111}–{111} edge; set equal to θ_110 (strongest blocking)

ALPHA_BCF = 0.20   # BCF edge/kink rate premium; tunable. Burton, Cabrera &
                   # Frank,(1951).Sensitivity sweep
                   # carried out: α = 0.10, 0.20, 0.30.
CN_FLOOR  = 3      # Reference site CN: flat {111} terrace hollow (exactly 3 atoms
                   # below). BCF premium = 0 here; ledge (CN=5) - 40% boost;
                   # kink (CN=7) - 80% boost.

# Base deposition rate constant k_dep (s⁻¹ per site) — scales with Au³⁺ solution
# concentration and reduction flux.  This is a fit/tunable parameter, not derived
# from DFT.  A reasonable starting value for dilute CTAB synthesis conditions:
K_DEP_BASE = 1e4   # s⁻¹ per unblocked site — adjusted during temperature sweep


def facet_from_cn(cn: int) -> str:
    """
    Map local coordination number to the most likely facet type.

    CN=9  - {111}  (close-packed terrace)
    CN=8  - {100}  (open terrace)
    CN≤7  - {110}  (step/edge — treated as {110}-like for CTAB blocking purposes)

    This is an approximation for the phenomenological CTAB model.
    """
    if cn >= 9:
        return '111'
    elif cn == 8:
        return '100'
    else:
        return '110'


_THETA = {'111': THETA_111, '100': THETA_100, '110': THETA_110}



def deposition_rate(cn: int,
                    k_dep: float = K_DEP_BASE,
                    is_silver_proximal: bool = False,
                    site_cn: int = CN_FLOOR) -> float:
    """
    Phenomenological deposition rate for a surface site.

    R = k_dep × (1 − θ_CTAB) × (1 − θ_Ag · is_silver_proximal) × f_bcf

    where f_bcf = 1 + ALPHA_BCF × max(0, site_cn − CN_FLOOR)

    BCF factor: under-coordinated sites (ledges, kinks) are stickier than
    flat terraces - more bonds form on landing, raising the effective rate.
    Burton, Cabrera & Frank, (1951).

    Deposition has no activation-barrier term. The rate is k_dep x site factors,
    where k_dep is a fixed phenomenological flux. It is held constant across temperatures,
    so temperature enters the model only through surface diffusion. Near-zero
    adsorption barriers were found in earlier NEB tests.

    Parameters
    ----------
    cn                 : int   — CN of the *parent* surface atom (3.0 Å cutoff);
                                 used for CTAB facet classification via facet_from_cn()
    k_dep              : float — base deposition rate constant, s⁻¹
    is_silver_proximal : bool  — True if parent atom is on a silver-stabilised
                                 {111}–{111} edge or its 3.0 Å neighbourhood
    site_cn            : int   — raw connectivity of the *candidate site* itself
                                 (atoms within CN_CUTOFF of the empty position);
                                 distinct from cn (parent). Default = CN_FLOOR = 3
                                 (flat terrace, no BCF premium).

    Returns
    -------
    float — effective deposition rate in s⁻¹.
    """
    facet = facet_from_cn(cn)
    theta = _THETA[facet]
    rate  = k_dep * (1.0 - theta)
    if is_silver_proximal:
        rate *= (1.0 - THETA_AG)
    f_bcf = 1.0 + ALPHA_BCF * max(0, site_cn - CN_FLOOR)
    rate *= f_bcf
    return rate


def deposition_rate_all_facets(k_dep: float = K_DEP_BASE) -> dict:
    """Convenience: return deposition rates for all three facet types."""
    return {f: k_dep * (1.0 - theta) for f, theta in _THETA.items()}
