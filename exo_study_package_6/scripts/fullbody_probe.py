"""Feasibility probe for the full-body model (myofullbody, with arms). Takes ~1-2 min. Does NOT train anything.
Usage (repo root, PYTHONPATH=.):
  python $P/make_configs.py --msk myofullbody --out work/fb_probe.json --payload_kg 20 --num_envs 1 --timesteps 1000
  python $P/fullbody_probe.py --config work/fb_probe.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import make_env
import mujoco
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--steps", type=int, default=100); a = ap.parse_args()
env = make_env(json.load(open(a.config)), payload_kg=20)
m = env.sim.model.ptr if hasattr(env.sim.model, "ptr") else env.sim.model
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
arm = [n for n in names if any(k in n for k in ("sh_", "shoulder", "elbow", "wrist", "arm"))]
print(f"actuators {m.nu} | joints {m.njnt} | bodies {m.nbody} | arm-related joints {len(arm)}: {arm[:8]}")
obs, _ = env.reset(); print("obs", obs.shape, "action", env.action_space.shape)
t = time.time(); n = 0
for _ in range(a.steps):
    obs, r, term, trunc, _ = env.step(env.action_space.sample() * 0.0); n += 1
    if term or trunc: break
dt = time.time() - t
print(f"{n} control steps in {dt:.1f} s -> {n/dt:.1f} steps/s (single env, zero muscle activation; episode ends when the model falls)")
