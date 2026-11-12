"""LOCAL PATCH (paper code): payload and virtual-population parameter application on a compiled MuJoCo model.
Payload is a point mass added to an existing body (no recompile): body mass, COM and inertia tensor are updated
(parallel-axis theorem). Approximation: the point mass has no own inertia. Verified only by unit test, not against hardware."""
import numpy as np
import mujoco

PAYLOAD_POS = {  # offset from the body frame origin, expressed in the body frame (m). ASSUMED geometry, not measured.
    "back": np.array([-0.12, 0.0, 0.0]),
    "front": np.array([0.15, 0.0, 0.0]),
}

def add_point_mass(model, body_name, mass, pos_body):
    model = getattr(model, "ptr", model)  # dm_control wrapper -> raw mujoco.MjModel
    b = model.body(body_name).id
    m0 = model.body_mass[b]
    ipos0 = model.body_ipos[b].copy()
    R = np.zeros(9); mujoco.mju_quat2Mat(R, model.body_iquat[b]); R = R.reshape(3, 3)
    I0 = R @ np.diag(model.body_inertia[b]) @ R.T          # inertia about own COM, body-frame axes
    M = m0 + mass
    com = (m0 * ipos0 + mass * pos_body) / M
    def shift(m, r):
        r = np.asarray(r); return m * (r @ r * np.eye(3) - np.outer(r, r))
    I = I0 + shift(m0, ipos0 - com) + shift(mass, pos_body - com)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0: V[:, 0] *= -1
    q = np.zeros(4); mujoco.mju_mat2Quat(q, V.flatten())
    model.body_mass[b] = M; model.body_ipos[b] = com
    model.body_inertia[b] = w; model.body_iquat[b] = q

def apply_payload(model, payload_kg, mode="back", body="torso"):
    if payload_kg <= 0: return
    model = getattr(model, "ptr", model)
    add_point_mass(model, body, payload_kg, PAYLOAD_POS[mode])
    mujoco.mj_setConst(model, mujoco.MjData(model))

def change_payload(env, delta_kg, mode="back", body="torso"):
    """Mid-episode load change (load step): adds delta_kg (>0) as a point mass, or removes it (<0) by adding a negative point mass
    (valid as long as the total torso mass stays positive). Uses the same point-mass approximation as apply_payload."""
    model = getattr(env.sim.model, "ptr", env.sim.model)
    add_point_mass(model, body, float(delta_kg), PAYLOAD_POS[mode])
    mujoco.mj_setConst(model, mujoco.MjData(model))

def apply_population(model, p, seed=0):
    """p: dict with mass_scale, seg_mass_dev, fmax_scale, len_shift, friction, exo_tau_scale (see population.py RANGES).
    delay_ms is NOT applied here (handled by the evaluation wrapper)."""
    model = getattr(model, "ptr", model)
    rng = np.random.default_rng(seed)
    for b in range(1, model.nbody):
        f = p["mass_scale"] * (1 + p["seg_mass_dev"] * rng.uniform(-1, 1))
        model.body_mass[b] *= f; model.body_inertia[b] *= f
    mus = np.where(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)[0]
    model.actuator_gainprm[mus, 2] *= p["fmax_scale"]; model.actuator_biasprm[mus, 2] *= p["fmax_scale"]
    sh = 1 + p["len_shift"]
    model.actuator_gainprm[mus, 0:2] *= sh; model.actuator_biasprm[mus, 0:2] *= sh
    model.geom_friction[:, 0] = p["friction"]
    for i in range(model.nu):
        if i not in mus: model.actuator_ctrlrange[i] = model.actuator_ctrlrange[i] * p["exo_tau_scale"]
    mujoco.mj_setConst(model, mujoco.MjData(model))


def setup_all(env, env_params):
    """Single entry point called from MyoAssistLegBase._setup (study patch): population -> payload -> assist overlay."""
    if getattr(env_params, "population_params", None):
        apply_population(env.sim.model, env_params.population_params, env_params.population_seed)
    apply_payload(env.sim.model, getattr(env_params, "payload_kg", 0.0),
                  getattr(env_params, "payload_mode", "back"), getattr(env_params, "payload_body", "torso"))
    from rl_train.envs.assist_overlay import setup_assist
    setup_assist(env, env_params)
