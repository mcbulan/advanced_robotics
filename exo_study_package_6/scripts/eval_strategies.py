"""Evaluate every strategy over payload x virtual individuals (+ nominal). Output: one CSV row per rollout.
The strategy's policy (fine-tuned with its overlay) and its alpha vector are loaded from sessions.json + strategies.json.
Device latency (delay_ms) and torque-limit scale (exo_tau_scale) of each individual act on the overlay; all other population
parameters change the musculoskeletal model. Usage (repo root):
  python scripts/eval_strategies.py --strategies configs/strategies.json --registry sessions.json --prefix ft_ --suffix _p20 \
      --payloads 10 20 30 --n_pop 16 --out results.csv"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from population import sample
ap = argparse.ArgumentParser(); ap.add_argument("--strategies", required=True); ap.add_argument("--registry", default="sessions.json")
ap.add_argument("--only", nargs="*", default=None, help="subset of strategy names"); ap.add_argument("--reg_prefix", default="ft_", help="registry key = prefix + strategy name + suffix")
ap.add_argument("--reg_suffix", default="_p20"); ap.add_argument("--payloads", type=float, nargs="+", default=[20.0])
ap.add_argument("--payload_mode", default="back"); ap.add_argument("--n_pop", type=int, default=16); ap.add_argument("--pop_seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--speed", type=float, default=1.25); ap.add_argument("--out", default="results.csv")
a = ap.parse_args()
S = json.load(open(a.strategies))["strategies"]; reg = json.load(open(a.registry))
names = a.only or list(S)
pops = [None] + sample(a.n_pop, seed=a.pop_seed)       # SAME individuals for every strategy
rows = []
for nm in names:
    sess = reg[a.reg_prefix + nm + a.reg_suffix]; model, cfg0 = load_policy(sess)
    for pk in a.payloads:
        for pi, p in enumerate(pops):
            asg = json.loads(json.dumps(S[nm])); asg.pop("budget_rho_target", None); asg.pop("budget_rho_realised", None)
            if p: asg["delay_ms"] = p["delay_ms"]; asg["tau_scale"] = p["exo_tau_scale"]
            env = make_env(cfg0, payload_kg=pk, payload_mode=a.payload_mode, population=p, pop_seed=pi, assist=asg)
            r = rollout(env, model, a.steps, a.speed); env.close()
            rows.append(dict(strategy=nm, payload_kg=pk, individual=(pi - 1) if p else "nominal", **r)); print(rows[-1]["strategy"], rows[-1]["payload_kg"], rows[-1]["individual"], round(r["survived_s"], 2), r["fell"], round(r["E_act"], 4), flush=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written", a.out)
