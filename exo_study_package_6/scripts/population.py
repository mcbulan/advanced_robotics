"""Virtual population: Latin-hypercube sampling of model-parameter uncertainty and application to a compiled MuJoCo model.
Ranges: see Table II of the paper (sources vs. ASSUMED). Fmax/length shifts act on the MuJoCo muscle gain/bias parameters
(tendon is rigid in this model, so tendon-compliance is NOT a parameter; lmin/lmax shift is its proxy)."""
import numpy as np
import mujoco
from scipy.stats import qmc

# name: (low, high, status)  status: 'sourced' (relative range taken from literature) or 'assumed'
RANGES = {
    "mass_scale":   (0.85, 1.15, "assumed"),   # global body-mass multiplier (anthropometric spread; no ANSUR II value verified)
    "seg_mass_dev": (-0.10, 0.10, "sourced"),  # per-segment independent mass error, +-10 % (Lewis 2019 / Chang 2018 range)
    "fmax_scale":   (0.75, 1.25, "sourced"),   # Blache 2019: +-25 %
    "len_shift":    (-0.05, 0.05, "sourced"),  # Astl 2026: 95-105 % (proxy for tendon slack length)
    "friction":     (0.5, 1.0, "assumed"),     # sliding friction mu
    "delay_ms":     (0.0, 40.0, "assumed"),    # control/perception delay (relative +0..40 % in Ryu & Koo 2026; absolute ms assumed)
    "exo_tau_scale": (0.75, 1.0, "assumed"),   # exo torque-limit scale (sensitivity axis)
}
NAMES = list(RANGES)

def sample(n, seed=0):
    """Scrambled Sobol/LHS samples as list of dicts (n should be a power of two for Sobol; LHS used here)."""
    lo = np.array([RANGES[k][0] for k in NAMES]); hi = np.array([RANGES[k][1] for k in NAMES])
    u = qmc.LatinHypercube(d=len(NAMES), seed=seed).random(n)
    x = qmc.scale(u, lo, hi)
    return [dict(zip(NAMES, row)) for row in x]

def apply(model, p, seed=0, orig=None):
    """Return a modified copy of `model` (original untouched)."""
    m = model  # caller passes a fresh compiled model per sample
    rng = np.random.default_rng(seed)
    for b in range(1, m.nbody):
        if m.body(b).name == "payload":
            continue
        f = p["mass_scale"] * (1 + p["seg_mass_dev"] * rng.uniform(-1, 1))
        m.body_mass[b] *= f; m.body_inertia[b] *= f
    ms = m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT  # exo actuators are joint-type, muscles are tendon-type
    mus = np.where(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)[0]
    m.actuator_gainprm[mus, 2] *= p["fmax_scale"]; m.actuator_biasprm[mus, 2] *= p["fmax_scale"]
    sh = 1 + p["len_shift"]
    m.actuator_gainprm[mus, 0:2] *= sh; m.actuator_biasprm[mus, 0:2] *= sh
    m.geom_friction[:, 0] = p["friction"]
    for i in range(m.nu):
        if i not in mus:
            lim = m.actuator_ctrlrange[i]; m.actuator_ctrlrange[i] = lim * p["exo_tau_scale"]
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m
