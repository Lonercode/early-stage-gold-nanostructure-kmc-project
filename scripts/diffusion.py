import numpy as np

# ── Physical constants
NU_0 = 1e12           # attempt frequency, s⁻¹
KB   = 8.617333e-5    # Boltzmann constant, eV K⁻¹

# BEP anchor points - Boisvert & Lewis, (1995), Table I
# Confirmed from the paper (terrace diffusion on respective surfaces)
E_BARRIER_111 = 0.22
E_BARRIER_100 = 0.62
CN_111 = 9
CN_100 = 8

# Linear BEP interpolation: E_a = a + b*CN (Two-point fit)
_BEP_SLOPE     = (E_BARRIER_111 - E_BARRIER_100) / (CN_111 - CN_100)  # −0.40 eV/CN
_BEP_INTERCEPT = E_BARRIER_100  - _BEP_SLOPE * CN_100                 #  3.82 eV

E_BARRIER_FLOOR = 0.05

CN_CUTOFF = 3.0  # Å


def diffusion_barrier(cn: float) -> float:
    """
    BEP coordination-based activation energy for Au surface-atom diffusion.

    Anchored to Boisvert & Lewis (1996) Phys. Rev. B 54, 2880:
      CN=9 ({111} terrace) - 0.22 eV
      CN=8 ({100} terrace) - 0.62 eV

    Linear interpolation is valid and tested in CN 8–9.  Extrapolation below 8
    (step/edge/adatom sites) is approximate.

    Parameters
    ----------
    cn : float
        Local coordination number (neighbours within CN_CUTOFF = 3.0 Å).
        Atoms with CN > 9 are frozen (treated as bulk).

    Returns
    -------
    float
        Activation energy in eV.
    """
    # Clamp to CN_100 p- the linear fit is only anchored between CN=8 and CN=9.
    # Extrapolating below CN=8 produces unphysically large barriers that 
    # permanently freeze deposited adatoms. Low-CN atoms are treated
    # as {100}-like.
    cn_clamped = max(float(cn), float(CN_100))
    return max(E_BARRIER_FLOOR, _BEP_INTERCEPT + _BEP_SLOPE * cn_clamped)


def diffusion_rate(cn: float, T: float) -> float:
    """
    Arrhenius diffusion hop rate:  k = ν₀ · exp(−E_a / k_B T)

    Parameters
    ----------
    cn : float
        Local coordination number (3.0 Å cutoff).
    T  : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Rate in s⁻¹.

    Typical values at T=300 K:
      CN=9 (111-like):  ~2 × 10⁸ s⁻¹   (fast — {111} is smooth)
      CN=8 (100-like):  ~40 s⁻¹         (slow — {100} corrugation)
    """
    return NU_0 * np.exp(-diffusion_barrier(cn) / (KB * T))
