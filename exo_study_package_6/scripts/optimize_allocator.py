"""Optimise the few allocator parameters (rho_slope, theta[3], phi[3]) with a simple evolution strategy (antithetic, common random numbers).
The policy is FIXED (unassisted ft_none policy); the true load is given to the allocator (oracle) so that the allocation law is tuned
independently of the estimator. Mean assistance budget is fixed: rho(L_ref)=rho_ref.
Cost per scenario:  J = -(1 - R_bio/R0) + w_E*(E_act/E0 - 1) + 5*fell + w_P*P_exo/100 + 5*viol_vc      (R0, E0: same scenario, no assistance)
Usage (repo root, PYTHONPATH=.):
  python scripts/optimize_allocator.py --session <ft_none_p20 session> --demand demand.json --limits limits.json --out alloc.json \
      --payloads 10 20 30 --n_ind 4 --gens 12 --pairs 6 --workers 8
Output alloc.json = dict(rho_ref, rho_slope, theta, phi, ...) used by make_learned_config.py. Values found are DESIGN values under the model."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HIP, KNEE, ANK = "hip", "knee", "ankle"

def assist_from(x, rho_ref, D, lim, L, umax):
    al = dict(rho_ref=rho_ref, rho_slope=float(x[0]), rho_max=0.6, load_ref=20.0, load_scale=10.0,
              theta=dict(hip=float(x[1]), knee=float(x[2]), ankle=float(x[3])), phi=dict(hip=float(x[4]), knee=float(x[5]), ankle=float(x[6])))
    return dict(controller="learned_qp", alpha=dict(hip=0, knee=0, ankle=0), umax=dict(hip=umax, knee=umax, ankle=umax), limits=lim,
                lc=dict(d_nominal=D, load_oracle_kg=float(L), alloc=al))

def run(job):
    session, model_path, x, rho_ref, D, lim, L, ind, steps, speed, umax, wE, wP = job
    import torch; torch.set_num_threads(1)
    from common import load_policy, make_env, rollout
    from population import sample
    model, cfg0 = load_policy(session, model_path); p = sample(ind + 1, seed=0)[ind]
    if x is None:
        asg = dict(alpha=dict(hip=0, knee=0, ankle=0), umax=dict(hip=umax, knee=umax, ankle=umax))
    else:
        asg = assist_from(x, rho_ref, D, lim, L, umax)
    asg["delay_ms"] = p["delay_ms"]; asg["tau_scale"] = p["exo_tau_scale"]
    env = make_env(cfg0, payload_kg=L, population=p, pop_seed=ind, assist=asg); r = rollout(env, model, steps, speed); env.close()
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--model", default=None)
    ap.add_argument("--demand", required=True); ap.add_argument("--limits", required=True); ap.add_argument("--out", default="alloc.json")
    ap.add_argument("--payloads", type=float, nargs="+", default=[10.0, 20.0, 30.0]); ap.add_argument("--n_ind", type=int, default=4)
    ap.add_argument("--gens", type=int, default=12); ap.add_argument("--pairs", type=int, default=6); ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--rho_ref", type=float, default=0.3); ap.add_argument("--wE", type=float, default=0.5); ap.add_argument("--wP", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--speed", type=float, default=1.25); ap.add_argument("--umax", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    Dd = json.load(open(a.demand)); D = dict(hip=Dd["D_hip"], knee=Dd["D_knee"], ankle=Dd["D_ankle"])
    lim = json.load(open(a.limits)); lim.pop("note", None)
    scen = [(L, i) for L in a.payloads for i in range(a.n_ind)]
    ex = ProcessPoolExecutor(a.workers)
    def jobs(x): return [(a.session, a.model, x, a.rho_ref, D, lim, L, i, a.steps, a.speed, a.umax, a.wE, a.wP) for L, i in scen]
    base = list(ex.map(run, jobs(None))); print("baseline (no assistance) done; falls:", sum(b["fell"] for b in base), flush=True)
    def cost(rs):
        J = []
        for r, b in zip(rs, base):
            J.append(-(1 - r["R_bio"] / b["R_bio"]) + a.wE * (r["E_act"] / b["E_act"] - 1) + 5 * r["fell"] + a.wP * r["P_exo"] / 100 + 5 * r.get("viol_vc", 0.0))
        return float(np.mean(J))
    rng = np.random.default_rng(a.seed); n = 7
    lo = np.array([0.0] + [-2.0] * 6); hi = np.array([0.15] + [2.0] * 6); sig = np.array([0.05] + [a.sigma] * 6)
    x = np.zeros(n); hist = []; best = (cost(list(ex.map(run, jobs(x)))), x.copy()); print("gen 0 J(x0) = %.4f" % best[0], flush=True)
    for g in range(1, a.gens + 1):
        eps = rng.standard_normal((a.pairs, n)) * sig; cand = []
        for e in eps: cand += [np.clip(x + e, lo, hi), np.clip(x - e, lo, hi)]
        allj = jobs(cand[0]); allj = [j for c in cand for j in jobs(c)]
        res = list(ex.map(run, allj)); k = len(scen); J = np.array([cost(res[i * k:(i + 1) * k]) for i in range(len(cand))])
        Jp, Jm = J[0::2], J[1::2]; grad = ((Jp - Jm)[:, None] * eps).sum(0) / (2 * a.pairs * (sig ** 2)) * (sig ** 2)   # antithetic gradient estimate (scaled by sig^2)
        x = np.clip(x - 0.5 * grad / (np.abs(grad).max() + 1e-9) * sig, lo, hi)      # normalised step of size 0.5 sigma
        j = cost(list(ex.map(run, jobs(x)))); hist.append(j)
        if j < best[0]: best = (j, x.copy())
        print("gen %d  J(x)=%.4f  best=%.4f  x=%s" % (g, j, best[0], np.round(x, 2).tolist()), flush=True)
    xb = best[1]; out = dict(rho_ref=a.rho_ref, rho_slope=float(xb[0]), rho_max=0.6, load_ref=20.0, load_scale=10.0,
        theta=dict(hip=float(xb[1]), knee=float(xb[2]), ankle=float(xb[3])), phi=dict(hip=float(xb[4]), knee=float(xb[5]), ankle=float(xb[6])))
    json.dump(out, open(a.out, "w"), indent=1); json.dump(dict(J_hist=hist, J_best=best[0], J_x0=cost(list(ex.map(run, jobs(np.zeros(n)))))), open(os.path.splitext(a.out)[0] + "_log.json", "w"), indent=1)
    print("saved", a.out, "| J_best %.4f" % best[0])
    if best[0] > -0.01: print("NOTE: optimisation found no clear gain over the default (demand-proportional) allocation; report this honestly.")

if __name__ == "__main__":
    main()
