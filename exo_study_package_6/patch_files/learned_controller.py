"""Learned load-aware torque allocator with a QP safety filter (study code, simulation only, unvalidated on hardware).

Three layers, called once per control step from AssistOverlay (mode  assist = {"controller": "learned_qp", ...}):
  1) SENSING   LoadEstimator : small MLP (numpy inference) maps window features (ground reaction, joint motion, joint moments)
                               to the carried load L_hat [kg]. Trained offline by scripts/train_estimator.py.
  2) ALLOCATION Allocator    : total assistance budget rho(L_hat) is split over hip/knee/ankle by weights w(L_hat) (softmax of
                               a few parameters, optimised by scripts/optimize_allocator.py) -> desired torque tau_des.
                               With default parameters (theta = phi = 0, rho_slope = 0) it equals the fixed "demand" strategy.
  3) SAFETY    QPFilter      : tau = argmin  (1/2)||tau - tau_des||^2 + (lam/2)||tau - tau_prev||^2
                               s.t.  lo_j <= tau_j <= hi_j   (actuator limit, rate limit, joint-limit margin = discrete barrier)
                                     sum_j |omega_j tau_j| <= P_max   (power budget)
                               and tau_des is scaled by g in [0,1] when the joint constraint-force indicator V_c nears its threshold.
                               The problem is separable except for the power row; it is solved exactly (soft-threshold + bisection).
All limit values are ASSUMED design parameters, not measured.
"""
import numpy as np
import mujoco

JOINTS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
GROUP_OF = np.array([0, 0, 1, 1, 2, 2])          # hip, knee, ankle
N_FEAT = 20
ALPHA_MAX = 1.0


# ---------------------------------------------------------------- QP safety filter
def qp_filter(tau_des, tau_prev, omega, lo, hi, pmax, lam=0.5):
    """Exact solution of the small QP above. Returns (tau, info). lo/hi: per-joint bounds (already include actuator, rate, margin)."""
    lo = np.minimum(lo, hi)
    t0 = np.clip((tau_des + lam * tau_prev) / (1.0 + lam), lo, hi)
    c = np.abs(omega)
    info = dict(power_active=0, box_active=int(np.any(np.abs(t0 - (tau_des + lam * tau_prev) / (1.0 + lam)) > 1e-9)), infeasible=0)
    if pmax is None or np.sum(c * np.abs(t0)) <= pmax:
        return t0, info
    info["power_active"] = 1
    free = (tau_des + lam * tau_prev) / (1.0 + lam)

    def sol(t):
        return np.clip(np.sign(free) * np.maximum(np.abs(free) - t * c, 0.0), lo, hi)

    a, b = 0.0, float(np.max(np.abs(free) / np.maximum(c, 1e-9))) + 1.0
    if np.sum(c * np.abs(sol(b))) > pmax:            # lower bounds force power above the cap: best effort
        info["infeasible"] = 1
        return sol(b), info
    for _ in range(40):
        m = 0.5 * (a + b)
        if np.sum(c * np.abs(sol(m))) > pmax: a = m
        else: b = m
    return sol(b), info


# ---------------------------------------------------------------- estimator
class LoadEstimator:
    def __init__(self, path=None):
        self.ok = False
        if path:
            z = np.load(path)
            self.W = [z["W1"], z["W2"], z["W3"]]; self.b = [z["b1"], z["b2"], z["b3"]]
            self.mu = z["mu"]; self.sd = z["sd"]; self.ok = True

    def predict(self, x):
        h = (x - self.mu) / self.sd
        h = np.tanh(h @ self.W[0] + self.b[0]); h = np.tanh(h @ self.W[1] + self.b[1])
        return float((h @ self.W[2] + self.b[2]).ravel()[0])


def save_estimator(path, torch_model, mu, sd):
    layers = [m for m in torch_model if hasattr(m, "weight")]
    d = {}
    for i, l in enumerate(layers, 1):
        d[f"W{i}"] = l.weight.detach().numpy().T.astype(np.float64); d[f"b{i}"] = l.bias.detach().numpy().astype(np.float64)
    np.savez(path, mu=mu, sd=sd, **d)


# ---------------------------------------------------------------- controller
DEFAULT_LC = dict(
    estimator_path=None, load_default=20.0, load_oracle_kg=None,       # oracle = ablation only (true load given to the controller)
    window=40, refresh=5, ema=0.8,
    d_nominal=dict(hip=20.0, knee=20.0, ankle=20.0),                   # fallback demand until the window is filled [Nm]
    alloc=dict(rho_ref=0.3, rho_slope=0.0, rho_max=0.6, load_ref=20.0, load_scale=10.0,
               theta=dict(hip=0.0, knee=0.0, ankle=0.0), phi=dict(hip=0.0, knee=0.0, ankle=0.0)),
    use_filter=True, lam=0.5, margin=0.15,                             # lam: smoothness weight; margin: joint-limit margin [rad]
)


def _merge(a, b):
    o = dict(a)
    for k, v in (b or {}).items():
        o[k] = _merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
    return o


class LearnedController:
    def __init__(self, model, cfg, umax, limits):
        self.c = _merge(DEFAULT_LC, cfg)
        self.m = getattr(model, "ptr", model)
        self.umax = np.asarray(umax, float); self.lim = limits           # dict dmax, pmax, vc_thr
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in JOINTS])
        self.jr = np.array([self.m.jnt_range[self.m.joint(n).id] for n in JOINTS])
        self.est = LoadEstimator(self.c["estimator_path"])
        self.floor_bodies = {0}
        self.reset()

    def reset(self):
        self.buf = []; self.L_hat = float(self.c["load_default"]); self.L_ema = None; self.n = 0
        self.D = np.array([self.c["d_nominal"]["hip"], self.c["d_nominal"]["knee"], self.c["d_nominal"]["ankle"]], float)
        self.tau_prev = np.zeros(6); self.vc_lp = 0.0; self.alpha = np.zeros(3); self.g = 1.0
        self.last = dict(power_active=0, box_active=0, infeasible=0)

    def _grf_kg(self, data):
        f6 = np.zeros(6); tot = 0.0
        for i in range(data.ncon):
            ct = data.contact[i]
            b1 = self.m.geom_bodyid[ct.geom1]; b2 = self.m.geom_bodyid[ct.geom2]
            if b1 in self.floor_bodies or b2 in self.floor_bodies:
                mujoco.mj_contactForce(self.m, data, i, f6); tot += max(f6[0], 0.0)
        return tot / 9.81

    def _features(self):
        B = np.array(self.buf[-self.c["window"]:])                     # rows: [grf, |w|x6, q x6, |tau_bio| x6]
        q = B[:, 7:13]
        return np.concatenate([[B[:, 0].mean()], B[:, 1:7].mean(0), q.max(0) - q.min(0), B[:, 13:19].mean(0)]), B

    def step(self, data, tau_bio, tau_lp, vc, dt):
        q = data.qpos[self.qadr]; w = data.qvel[self.dadr]
        self.buf.append(np.concatenate([[self._grf_kg(data)], np.abs(w), q, np.abs(tau_bio)])); self.buf = self.buf[-self.c["window"]:]
        self.n += 1
        # 1) sensing
        if self.c["load_oracle_kg"] is not None:
            self.L_hat = float(self.c["load_oracle_kg"])
        elif len(self.buf) >= self.c["window"] // 2 and self.n % self.c["refresh"] == 0:
            x, B = self._features()
            if self.est.ok:
                raw = float(np.clip(self.est.predict(x), 0.0, 40.0))
                self.L_ema = raw if self.L_ema is None else self.c["ema"] * self.L_ema + (1 - self.c["ema"]) * raw
                self.L_hat = self.L_ema
        if len(self.buf) >= self.c["window"] // 2:
            tb = np.array(self.buf)[:, 13:19].mean(0)
            self.D = np.array([tb[GROUP_OF == g].mean() for g in range(3)])
        # 2) allocation
        A = self.c["alloc"]; z = (self.L_hat - A["load_ref"]) / A["load_scale"]
        rho = float(np.clip(A["rho_ref"] + A["rho_slope"] * z, 0.0, A["rho_max"]))
        lg = np.array([A["theta"][k] + A["phi"][k] * z for k in ("hip", "knee", "ankle")])
        wgt = np.exp(lg - lg.max()); wgt /= wgt.sum()
        S = float(self.D.sum()); Dg = np.maximum(self.D, 1e-3)
        self.alpha = np.clip(rho * S * wgt / Dg, 0.0, ALPHA_MAX)
        tau_des = self.alpha[GROUP_OF] * tau_lp
        # 3) safety
        self.vc_lp = 0.8 * self.vc_lp + 0.2 * vc
        thr = self.lim.get("vc_thr")
        self.g = 1.0 if not thr else float(np.clip((thr - self.vc_lp) / (0.3 * thr), 0.0, 1.0))
        tau_des = np.clip(self.g * tau_des, -self.umax, self.umax)
        if not self.c["use_filter"]:
            out = tau_des; self.last = dict(power_active=0, box_active=0, infeasible=0)
        else:
            dmax = self.lim.get("dmax", 60.0); mg = self.c["margin"]
            hi_s = np.clip((self.jr[:, 1] - q) / mg, 0.0, 1.0); lo_s = np.clip((q - self.jr[:, 0]) / mg, 0.0, 1.0)
            lo = np.maximum(-self.umax * lo_s, self.tau_prev - dmax); hi = np.minimum(self.umax * hi_s, self.tau_prev + dmax)
            out, self.last = qp_filter(tau_des, self.tau_prev, w, lo, hi, self.lim.get("pmax"), self.c["lam"])
        self.tau_prev = out.copy()
        return out
