"""Load-step evaluation: payload changes during walking (L0 -> L1 at step --step_at of --steps). Controllers:
  none (ft_none policy) | demand (fixed demand-proportional, ft_demand policy) | learned_nofilter | learned_qp | learned_oracle  (all three: ft_learned policy)
Same virtual individuals for every controller. Output CSV: one row per rollout. Usage (repo root, PYTHONPATH=.):
  python scripts/eval_learned.py --registry sessions.json --strategies configs/strategies.json --learned_assist work/ft_learned_p20_assist.json \
     --steps_list 10:25 25:10 --n_ind 8 --workers 8 --out eval_learned.csv [--reg_suffix _p20] [--learned_key ft_learned_p20]
Metrics: Lhat_err_end (|L_hat - L1|, mean over last 90 steps), settle_s (time after step until |L_hat-L1|<=3 kg for 30 consecutive steps),
tauBio_pre/post (mean |net muscle moment| before/after the step), E_act, violation rates, fell."""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CTRL = ["none", "demand", "learned_nofilter", "learned_qp", "learned_oracle"]

def run(job):
    name, reg, strat, lassist, key_l, suffix, L0, L1, ind, steps, step_at, speed = job
    import torch; torch.set_num_threads(1)
    from common import load_policy, make_env, rollout
    from population import sample
    from rl_train.envs.env_perturbations import change_payload
    p = sample(ind + 1, seed=0)[ind]
    if name in ("none", "demand"):
        sess = reg["ft_" + name + suffix]; asg = json.loads(json.dumps(strat[name])); asg.pop("budget_rho_target", None); asg.pop("budget_rho_realised", None)
    else:
        sess = reg[key_l]; asg = json.loads(json.dumps(lassist))
        if name == "learned_nofilter": asg["lc"]["use_filter"] = False
        if name == "learned_oracle": asg["lc"]["load_oracle_kg"] = float(L0)
        asg["lc"].pop("load_oracle_kg", None) if name != "learned_oracle" else None
    asg["delay_ms"] = p["delay_ms"]; asg["tau_scale"] = p["exo_tau_scale"]
    model, cfg0 = load_policy(sess); env = make_env(cfg0, payload_kg=L0, population=p, pop_seed=ind, assist=asg)
    def hook(i, e):
        if i == step_at:
            change_payload(e, L1 - L0)
            if name == "learned_oracle": e._assist.lc.c["load_oracle_kg"] = float(L1)
    r, tr = rollout(env, model, steps, speed, hook=hook, return_trace=True); env.close()
    row = dict(controller=name, L0=L0, L1=L1, individual=ind, fell=r["fell"], survived_s=r["survived_s"], E_act=r["E_act"], P_exo=r["P_exo"],
               viol_rate=r.get("viol_rate", 0.0), viol_power=r.get("viol_power", 0.0), viol_vc=r.get("viol_vc", 0.0), V_c=r["V_c"])
    tb = np.array([t["tauBio"] for t in tr]); lh = np.array([t["L_hat"] for t in tr]); dt = tr[0]["t"] if tr else 0.0
    row["tauBio_pre"] = float(tb[max(0, step_at - 90):step_at].mean()) if len(tb) > step_at else float("nan")
    row["tauBio_post"] = float(tb[-90:].mean()) if len(tb) >= steps else float("nan")
    if name.startswith("learned") and len(lh) >= steps:
        row["Lhat_err_pre"] = float(np.abs(lh[max(0, step_at - 90):step_at] - L0).mean()); row["Lhat_err_end"] = float(np.abs(lh[-90:] - L1).mean())
        ok = np.abs(lh[step_at:] - L1) <= 3.0; run_ = 0; st = float("nan")
        for k, v in enumerate(ok):
            run_ = run_ + 1 if v else 0
            if run_ >= 30: st = (k - 29) * dt; break
        row["settle_s"] = st
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--registry", default="sessions.json"); ap.add_argument("--strategies", required=True)
    ap.add_argument("--learned_assist", required=True); ap.add_argument("--reg_suffix", default="_p20"); ap.add_argument("--learned_key", default="ft_learned_p20")
    ap.add_argument("--steps_list", nargs="+", default=["10:25", "25:10"]); ap.add_argument("--n_ind", type=int, default=8)
    ap.add_argument("--steps", type=int, default=450); ap.add_argument("--step_at", type=int, default=150); ap.add_argument("--speed", type=float, default=1.25)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default="eval_learned.csv"); a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    reg = json.load(open(a.registry)); strat = json.load(open(a.strategies))["strategies"]; lassist = json.load(open(a.learned_assist))
    names = a.only or CTRL; jobs = []
    for s in a.steps_list:
        L0, L1 = [float(v) for v in s.split(":")]
        for nm in names:
            for i in range(a.n_ind): jobs.append((nm, reg, strat, lassist, a.learned_key, a.reg_suffix, L0, L1, i, a.steps, a.step_at, a.speed))
    with ProcessPoolExecutor(a.workers) as ex: rows = list(ex.map(run, jobs))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("controller", "L0", "L1", "individual"), k))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    import collections
    print("%-18s %-8s %5s %8s %8s %8s %8s %8s" % ("controller", "L0:L1", "falls", "Lh_err", "settle_s", "help_pre", "help_post", "viol_vc"))
    base = {(r["L0"], r["L1"], r["individual"]): r for r in rows if r["controller"] == "none"}
    for s in a.steps_list:
        L0, L1 = [float(v) for v in s.split(":")]
        for nm in names:
            R = [r for r in rows if r["controller"] == nm and r["L0"] == L0 and r["L1"] == L1]
            def mean(f):
                v = [f(r) for r in R]; v = [x for x in v if x == x and x is not None]; return float(np.mean(v)) if v else float("nan")
            hp = lambda k: (lambda r: 1 - r[k] / base[(r["L0"], r["L1"], r["individual"])][k] if (r["L0"], r["L1"], r["individual"]) in base else float("nan"))
            print("%-18s %-8s %5d %8.2f %8.2f %8.3f %8.3f %8.4f" % (nm, s, sum(r["fell"] for r in R), mean(lambda r: r.get("Lhat_err_end", float("nan"))),
                  mean(lambda r: r.get("settle_s", float("nan"))), mean(hp("tauBio_pre")), mean(hp("tauBio_post")), mean(lambda r: r["viol_vc"])))
    print("written", a.out)

if __name__ == "__main__":
    main()
