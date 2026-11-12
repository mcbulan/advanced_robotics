"""Gate 1 check: does the base policy walk? Deterministic rollouts on the nominal model (+ optional virtual individuals).
--device on  (DEFAULT) = exactly as trained/as in the training videos (policy also drives the Hippo exo).
--device off = exo forced to zero torque (the base policy was NOT trained this way, so falling here is expected).
Usage (repo root): python exo_study_package/scripts/check_gate1.py --session <session_dir> --payload_kg 20 [--steps 600] [--n_pop 3]
PASS (heuristic, ASSUMED thresholds): survived >= 10 s, no fall, mean speed within +-0.35 m/s of target. Walking QUALITY (gait pattern) is not judged here."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from population import sample
ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--model", default=None)
ap.add_argument("--payload_kg", type=float, default=0.0); ap.add_argument("--payload_mode", default="back")
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--speed", type=float, default=1.25)
ap.add_argument("--n_pop", type=int, default=0); ap.add_argument("--device", choices=["on", "off"], default="on"); a = ap.parse_args()
model, cfg0 = load_policy(a.session, a.model)
print("model:", a.model or latest_model(a.session), "| device:", a.device)
rows = []
for i, p in enumerate([None] + sample(a.n_pop, seed=0) if a.n_pop else [None]):
    env = make_env(cfg0, payload_kg=a.payload_kg, payload_mode=a.payload_mode, population=p, pop_seed=i, assist=None if a.device == "on" else dict(alpha=dict(hip=0, knee=0, ankle=0)))
    r = rollout(env, model, a.steps, a.speed)
    ok = (not r["fell"]) and r["survived_s"] >= 10.0 and abs(r["mean_speed"] - a.speed) <= 0.35
    rows.append(ok)
    print(f"{'nominal' if p is None else 'ind'+str(i-1):8s} survived {r['survived_s']:5.1f} s  fell {r['fell']}  speed {r['mean_speed']:.2f} m/s (target {a.speed})  dist {r['distance_m']:.1f} m  "
          f"E_act {r['E_act']:.3f}  R_bio {r['R_bio']:.3f}  -> {'PASS' if ok else 'FAIL'}")
print("GATE 1 (nominal):", "PASS" if rows[0] else "FAIL", "| individuals passed:", f"{sum(rows[1:])}/{len(rows)-1}" if len(rows) > 1 else "n/a")
