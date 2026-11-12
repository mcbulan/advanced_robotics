"""Measure the biological joint-moment demand D_j (mean |net muscle moment|, Nm; hip/knee/ankle, both legs averaged)
of a trained policy WITHOUT assistance (alpha = 0) on the nominal individual. Output: demand.json.
Usage: python scripts/measure_demand.py --session <dir> --payload_kg 20 --out demand.json"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--model", default=None)
ap.add_argument("--payload_kg", type=float, default=0.0); ap.add_argument("--payload_mode", default="back")
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--speed", type=float, default=1.25)
ap.add_argument("--out", default="demand.json"); a = ap.parse_args()
model, cfg0 = load_policy(a.session, a.model)
env = make_env(cfg0, payload_kg=a.payload_kg, payload_mode=a.payload_mode, assist=dict(alpha=dict(hip=0, knee=0, ankle=0)))
r = rollout(env, model, a.steps, a.speed)
g = lambda s: 0.5 * (r["tauBio_" + s + "_r"] + r["tauBio_" + s + "_l"])
d = dict(D_hip=g("hip_flexion"), D_knee=g("knee_angle"), D_ankle=g("ankle_angle"), payload_kg=a.payload_kg,
         survived_s=r["survived_s"], fell=r["fell"], mean_speed=r["mean_speed"])
print(json.dumps(d, indent=1))
if r["fell"] or r["survived_s"] < 8.0:
    print("WARNING: the unassisted policy did not walk >= 8 s; demand values are not meaningful (Gate 1 not passed).")
json.dump(d, open(a.out, "w"), indent=1)
