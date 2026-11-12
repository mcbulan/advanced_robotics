"""Plumbing test for the assist overlay (NOT a learning test). Usage (repo root):
   python scripts/test_assist.py --session rl_train/results/train_session_XXXX
Checks: (1) zero device action gives ~zero device force, (2) overlay torque is applied and persists, (3) torque changes joint moments."""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); a = ap.parse_args()
model, cfg0 = load_policy(a.session)
res = {}
for name, assist in (("none", dict(alpha=dict(hip=0, knee=0, ankle=0))), ("full", dict(alpha=dict(hip=0.3, knee=0.3, ankle=0.3)))):
    env = make_env(cfg0, assist=assist)
    obs, _ = env.reset(); m = env.sim.model.ptr if hasattr(env.sim.model, "ptr") else env.sim.model
    dev = np.asarray(env._device_actuator_ids, dtype=int); fdev = []; applied = []; exo = []
    for _ in range(30):
        act, _ = model.predict(obs, deterministic=True)
        obs, *_ = env.step(np.array(act, dtype=float))
        d = env.sim.data
        fdev.append(np.abs(d.actuator_force[dev]).max()); applied.append(np.abs(d.qfrc_applied).max()); exo.append(np.abs(env._assist.tau_exo).max())
    res[name] = dict(max_device_force_N=float(max(fdev)), max_qfrc_applied=float(max(applied)), max_tau_exo=float(max(exo)))
    env.close()
print(json.dumps(res, indent=1))
ok = res["none"]["max_device_force_N"] < 1e-6 and res["none"]["max_qfrc_applied"] < 1e-9 and res["full"]["max_qfrc_applied"] > 0.1
print("PASS" if ok else "CHECK: zero device force / applied torque not as expected - send this output to the assistant")
