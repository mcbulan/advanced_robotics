"""Suggest ASSUMED constraint values from two nominal rollouts (design choice, not a measured physical limit):
  dmax   = tighten * 99th percentile of |torque change per step| of the 'demand' strategy
  pmax   = tighten * 95th percentile of total exo power of the 'demand' strategy
  vc_thr = 1.25 * 95th percentile of the joint constraint-force indicator V_c of the unassisted policy
Usage: python scripts/calibrate_limits.py --strategies configs/strategies.json --registry sessions.json --payload_kg 20 --out limits.json [--reg_suffix _p20]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--strategies", required=True); ap.add_argument("--registry", default="sessions.json")
ap.add_argument("--payload_kg", type=float, default=20.0); ap.add_argument("--reg_suffix", default="_p20"); ap.add_argument("--tighten", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--speed", type=float, default=1.25); ap.add_argument("--out", default="limits.json"); a = ap.parse_args()
S = json.load(open(a.strategies))["strategies"]; reg = json.load(open(a.registry)); R = {}
for nm in ("none", "demand"):
    model, cfg0 = load_policy(reg["ft_" + nm + a.reg_suffix]); asg = json.loads(json.dumps(S[nm])); asg.pop("budget_rho_target", None); asg.pop("budget_rho_realised", None)
    env = make_env(cfg0, payload_kg=a.payload_kg, assist=asg); R[nm] = rollout(env, model, a.steps, a.speed); env.close()
    print(nm, {k: round(v, 3) for k, v in R[nm].items() if k in ("survived_s", "fell", "P_p95", "P_max", "rate_p99", "V_c_p95")})
lim = dict(dmax=a.tighten * R["demand"]["rate_p99"], pmax=a.tighten * R["demand"]["P_p95"], vc_thr=1.25 * R["none"]["V_c_p95"], note="ASSUMED design values from calibrate_limits.py")
json.dump(lim, open(a.out, "w"), indent=1); print(json.dumps(lim, indent=1))
if R["demand"]["fell"] or R["none"]["fell"]: print("WARNING: a calibration rollout fell; values are not meaningful.")
