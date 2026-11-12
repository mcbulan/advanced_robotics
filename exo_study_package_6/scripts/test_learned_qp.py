"""Plumbing/unit test for the learned controller (NOT a learning test). Usage (repo root, PYTHONPATH=.):
   python scripts/test_learned_qp.py [--session rl_train/results/train_session_XXXX]
Test 1 (no simulator): the QP filter is compared with a general-purpose solver (scipy SLSQP) on random problems.
Test 2 (needs --session): the learned_qp mode runs inside the environment, constraints are respected, and a mid-episode load step works."""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--session", default=None); a = ap.parse_args()

# ---- test 1: QP vs SLSQP
from scipy.optimize import minimize
try: from rl_train.envs.learned_controller import qp_filter
except Exception: sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch_files")); from learned_controller import qp_filter
rng = np.random.default_rng(0); worst = 0.0; nact = 0; lam = 0.5
for k in range(200):
    des = rng.uniform(-60, 60, 6); prev = rng.uniform(-40, 40, 6); om = rng.uniform(-6, 6, 6)
    lo = np.maximum(-rng.uniform(20, 60, 6), prev - 30); hi = np.minimum(rng.uniform(20, 60, 6), prev + 30); pmax = rng.uniform(20, 150)
    t, info = qp_filter(des, prev, om, lo, hi, pmax, lam); nact += info["power_active"]
    if info["infeasible"]: continue
    # SLSQP on the same QP with auxiliary variables p_j >= |om_j tau_j|, sum p <= pmax
    f = lambda z: 0.5 * np.sum((z[:6] - des) ** 2) + 0.5 * lam * np.sum((z[:6] - prev) ** 2)
    cons = [dict(type="ineq", fun=lambda z: z[6:] - om * z[:6]), dict(type="ineq", fun=lambda z: z[6:] + om * z[:6]), dict(type="ineq", fun=lambda z: pmax - z[6:].sum())]
    z0 = np.concatenate([np.clip(prev, lo, hi), np.abs(om * np.clip(prev, lo, hi))])
    r = minimize(f, z0, constraints=cons, bounds=[(l, h) for l, h in zip(np.minimum(lo, hi), hi)] + [(0, None)] * 6, method="SLSQP", options=dict(maxiter=300, ftol=1e-10))
    worst = max(worst, abs(f(np.concatenate([t, np.zeros(6)])) - r.fun))
    assert np.sum(np.abs(om * t)) <= pmax * (1 + 1e-6) + 1e-6 and np.all(t >= np.minimum(lo, hi) - 1e-9) and np.all(t <= hi + 1e-9)
print(f"test 1: constraints satisfied on 200 random problems ({nact} with active power row); worst cost gap vs SLSQP = {worst:.2e}")
print("test 1:", "PASS" if worst < 1e-3 else "CHECK (cost gap large)")
if not a.session: sys.exit(0)

# ---- test 2: inside the environment
from common import *
model, cfg0 = load_policy(a.session)
lim = dict(dmax=15.0, pmax=150.0, vc_thr=None)
asg = dict(controller="learned_qp", umax=dict(hip=60, knee=60, ankle=60), limits=lim,
           lc=dict(load_default=20.0, alloc=dict(rho_ref=0.6), d_nominal=dict(hip=30, knee=30, ankle=30)))
env = make_env(cfg0, payload_kg=10.0, assist=asg)
from rl_train.envs.env_perturbations import change_payload
def hook(i, e):
    if i == 10: change_payload(e, 15.0)
r, tr = rollout(env, model, 40, 1.25, hook=hook, return_trace=True); env.close()
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("survived_s", "fell", "P_exo", "P_max", "rate_p99", "viol_rate", "viol_power", "L_hat_end")})
print("mass before/after load step:", round(tr[5]["mass"], 2), round(tr[-1]["mass"], 2) if len(tr) > 12 else "(episode ended before step)")
ok = r["viol_rate"] == 0 and r["viol_power"] == 0 and r["P_max"] <= lim["pmax"] * 1.01
if len(tr) > 12: ok = ok and abs(tr[-1]["mass"] - tr[5]["mass"] - 15.0) < 1e-6
print("test 2:", "PASS" if ok else "CHECK: send this output to the assistant")
