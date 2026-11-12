"""Assist overlay (study code): ideal joint-torque exoskeleton applied on hip/knee/ankle via data.qfrc_applied.

Law (measured-signal based, NOT an oracle):  tau_exo_j = clip(alpha_j * LP(tau_bio_j), +-umax_j * tau_scale)
  tau_bio_j = net muscle moment at joint j from the previous step (data.qfrc_actuator), LP = first-order low-pass.
alpha_j      : per-joint assistance ratio (the "torque allocation strategy" is the vector alpha).
Device latency is applied to the command as a fractional delay (linear interpolation between control steps).
The overlay is an IDEAL actuator model: no mass, no interface compliance, no bandwidth beyond the LP/delay.
Everything here is simulation-only and unvalidated against hardware.

Optional mode  assist["controller"] = "learned_qp"  (see learned_controller.py): the fixed vector alpha is replaced by a learned,
load-aware allocation followed by a QP safety filter. assist["limits"] = dict(dmax, pmax, vc_thr) sets the constraint values
(ASSUMED); violations of these limits are counted for EVERY mode so that fixed strategies and the learned controller are comparable."""
import numpy as np
import mujoco

JOINTS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
GROUP = {"hip": [0, 1], "knee": [2, 3], "ankle": [4, 5]}
DEFAULT = dict(alpha=dict(hip=0.0, knee=0.0, ankle=0.0), umax=dict(hip=60.0, knee=60.0, ankle=60.0),  # Nm, ASSUMED
               lp=0.5, delay_ms=0.0, tau_scale=1.0, zero_device=True,
               controller=None, lc=None, limits=dict(dmax=60.0, pmax=None, vc_thr=None))  # dmax: max torque change per control step [Nm]


class AssistOverlay:
    def __init__(self, model, cfg):
        c = {**DEFAULT, **(cfg or {})}
        c["alpha"] = {**DEFAULT["alpha"], **c.get("alpha", {})}
        c["umax"] = {**DEFAULT["umax"], **c.get("umax", {})}
        self.cfg = c
        m = getattr(model, "ptr", model)
        self.dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.alpha = np.zeros(6); self.umax = np.zeros(6)
        for g, idx in GROUP.items():
            self.alpha[idx] = c["alpha"][g]; self.umax[idx] = c["umax"][g] * c["tau_scale"]
        self.lp = float(c["lp"]); self.delay_s = c["delay_ms"] / 1000.0
        self.limits = {**DEFAULT["limits"], **(c.get("limits") or {})}
        self.lc = None
        if c.get("controller") == "learned_qp":
            from rl_train.envs.learned_controller import LearnedController
            self.lc = LearnedController(model, c.get("lc"), self.umax, self.limits)
        self.reset(0.0)

    def reset(self, t):
        self.tau_f = np.zeros(6); self.hist = []; self.last_t = t
        self.tau_exo = np.zeros(6); self.tau_bio = np.zeros(6); self.power = 0.0
        self.rate = 0.0; self.vc = 0.0; self.nstep = 0; self.nviol = dict(rate=0, power=0, vc=0)
        self.prev_out = np.zeros(6)
        if getattr(self, "lc", None) is not None: self.lc.reset()

    def apply(self, data, dt, env=None):
        if data.time < self.last_t - 1e-9:      # episode reset detected
            self.reset(data.time)
        self.last_t = data.time
        self.tau_bio = data.qfrc_actuator[self.dofs].copy()
        self.tau_f = self.lp * self.tau_f + (1 - self.lp) * self.tau_bio
        self.vc = 0.0
        if env is not None:
            mm = getattr(env.sim.model, "ptr", env.sim.model)
            self.vc = float(env._get_max_joint_constraint_force() / (mm.body_mass.sum() * 9.81))
        if self.lc is not None:
            cmd = self.lc.step(data, self.tau_bio, self.tau_f, self.vc, dt)
        else:
            cmd = np.clip(self.alpha * self.tau_f, -self.umax, self.umax)
        self.hist.append(cmd); self.hist = self.hist[-4:]
        d = self.delay_s / dt; k = int(np.floor(d)); f = d - k
        get = lambda j: self.hist[max(len(self.hist) - 1 - j, 0)]
        out = (1 - f) * get(k) + f * get(k + 1)
        data.qfrc_applied[self.dofs] = out
        self.tau_exo = out
        self.power = float(np.sum(np.abs(out * data.qvel[self.dofs])))   # W, absolute (no regeneration credit)
        self.rate = float(np.max(np.abs(out - self.prev_out))); self.prev_out = out.copy(); self.nstep += 1
        L = self.limits
        self.nviol["rate"] += int(self.rate > L["dmax"] * 1.001)
        self.nviol["power"] += int(L.get("pmax") is not None and self.power > L["pmax"] * 1.001)
        self.nviol["vc"] += int(L.get("vc_thr") is not None and self.vc > L["vc_thr"])


def setup_assist(env, env_params):
    cfg = getattr(env_params, "assist", None)
    env._assist = AssistOverlay(env.sim.model, cfg) if cfg else None


def assist_pre_step(env, a):
    """Called at the start of MyoAssistLegBase.step. Returns the (possibly modified) action."""
    ov = getattr(env, "_assist", None)
    if ov is None:
        return a
    ov.apply(env.sim.data.ptr if hasattr(env.sim.data, "ptr") else env.sim.data, env.dt, env)
    if ov.cfg["zero_device"]:
        a = np.array(a, dtype=float, copy=True)
        dev = getattr(env, "_device_actuator_ids", None)
        if dev is not None and len(dev):
            n_mus = env.sim.model.nu - len(dev)
            a[n_mus:] = ZERO_ACTION_VALUE
    return a

ZERO_ACTION_VALUE = 0.0   # checked by scripts/test_assist.py: must give zero device force
