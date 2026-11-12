"""Collect training data for the load estimator: the unassisted policy walks with random carried load (0-30 kg), random virtual
individuals and (optionally) a random mid-episode load change. Features are computed exactly as inside the controller.
Usage (repo root, PYTHONPATH=.):
  python scripts/collect_estimator_data.py --session <ft_none_p20 session> --n_rollouts 160 --workers 8 --out est_data.npz [--slope_deg 8 is NOT needed here]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def work(args):
    session, model_path, seeds, steps, speed, lo, hi, p_step = args
    import torch; torch.set_num_threads(1)
    from common import load_policy, make_env, rollout
    from population import sample
    from rl_train.envs.env_perturbations import change_payload
    model, cfg0 = load_policy(session, model_path); X, Y, G = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd); p = sample(1, seed=int(sd))[0]; L = float(rng.uniform(lo, hi)); L0 = L
        step_at = int(rng.integers(steps // 4, 3 * steps // 4)) if rng.random() < p_step else None; newL = float(rng.uniform(lo, hi))
        asg = dict(controller="learned_qp", alpha=dict(hip=0, knee=0, ankle=0), delay_ms=p["delay_ms"], tau_scale=p["exo_tau_scale"],
                   lc=dict(use_filter=False, alloc=dict(rho_ref=0.0, rho_max=0.0)))        # rho = 0 -> no torque = unassisted
        env = make_env(cfg0, payload_kg=L, population=p, pop_seed=int(sd), assist=asg)
        state = dict(L=L, last_step=-999)
        def hook(i, e):
            lc = e._assist.lc
            if step_at is not None and i == step_at:
                change_payload(e, newL - state["L"]); state["L"] = newL; state["last_step"] = i
            if len(lc.buf) >= lc.c["window"] // 2 and (i % 5 == 0) and i - state["last_step"] > lc.c["window"] // 2 + 5:
                x, _ = lc._features(); X.append(x); Y.append(state["L"]); G.append(sd)
        rollout(env, model, steps, speed, hook=hook); env.close()
    return np.array(X), np.array(Y), np.array(G)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); ap.add_argument("--model", default=None)
    ap.add_argument("--n_rollouts", type=int, default=160); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--speed", type=float, default=1.25); ap.add_argument("--lo", type=float, default=0.0); ap.add_argument("--hi", type=float, default=30.0)
    ap.add_argument("--p_step", type=float, default=0.5); ap.add_argument("--out", default="est_data.npz"); a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    chunks = [list(range(w, a.n_rollouts, a.workers)) for w in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, [(a.session, a.model, c, a.steps, a.speed, a.lo, a.hi, a.p_step) for c in chunks]))
    X = np.concatenate([r[0] for r in res if len(r[0])]); Y = np.concatenate([r[1] for r in res if len(r[1])]); G = np.concatenate([r[2] for r in res if len(r[2])])
    np.savez(a.out, X=X, y=Y, g=G); print("samples", len(Y), "rollouts", len(set(G.tolist())), "->", a.out)
    if len(Y) < 500: print("WARNING: few samples; the policy probably falls early at some loads (check Gate 1 and the payload range).")
