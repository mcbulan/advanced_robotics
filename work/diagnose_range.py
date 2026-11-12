"""Run from the MyoAssist repository root. Finds the payload-step range tolerated by each controller (nominal and virtual individuals)."""
import copy
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.environ.get("P", "exo_study_package_6/scripts")))
sys.path.insert(0, os.path.abspath("."))

import argparse
STEPS, STEP_AT, SPEED = 450, 150, 1.25
N_IND = 6
# (scenario name, L0, L1, use_population)
SCENARIOS = []


def run(job):
    import torch
    torch.set_num_threads(1)
    from common import load_policy, make_env, rollout
    from population import sample
    from rl_train.envs.env_perturbations import change_payload

    name, key, assist0, scen, L0, L1, use_pop, ind = job
    assist = copy.deepcopy(assist0)
    assist.pop("budget_rho_target", None)
    assist.pop("budget_rho_realised", None)
    p = None
    if use_pop:
        p = sample(ind + 1, seed=0)[ind]
        assist["delay_ms"] = p["delay_ms"]
        assist["tau_scale"] = p["exo_tau_scale"]
    with open("work/sessions.json", encoding="utf-8") as handle:
        registry = json.load(handle)
    model, config = load_policy(registry[key])
    env = make_env(config, payload_kg=L0, population=p, pop_seed=ind, assist=assist)

    def hook(i, e):
        if i == STEP_AT and L1 != L0:
            change_payload(e, L1 - L0)

    try:
        r = rollout(env, model, STEPS, SPEED, hook=hook)
    finally:
        env.close()
    return name, scen, ind, int(r["fell"]), float(r["survived_s"])


def main():
    global SCENARIOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_list", nargs="+", default=["20:25", "25:20", "20:15", "15:20"])
    ap.add_argument("--n_ind", type=int, default=6)
    a = ap.parse_args()
    for item in a.steps_list:
        l0, l1 = [float(v) for v in item.split(":")]
        SCENARIOS.append((f"nominal_{item}", l0, l1, False))
        SCENARIOS.append((f"birey_{item}", l0, l1, True))
    global N_IND
    N_IND = a.n_ind
    with open("work/cfg/strategies.json", encoding="utf-8") as handle:
        strategies = json.load(handle)["strategies"]
    with open("work/ft_learned_p20_assist.json", encoding="utf-8") as handle:
        learned = json.load(handle)
    cases = [
        ("none", "ft_none_p20", strategies["none"]),
        ("demand", "ft_demand_p20", strategies["demand"]),
        ("learned_qp", "ft_learned_p20", learned),
    ]
    jobs = []
    for name, key, assist in cases:
        for scen, l0, l1, use_pop in SCENARIOS:
            for ind in (range(N_IND) if use_pop else [0]):
                jobs.append((name, key, assist, scen, l0, l1, use_pop, ind))
    with Pool(6) as pool:
        results = pool.map(run, jobs, chunksize=1)
    for name, _, _ in cases:
        for scen, *_ in SCENARIOS:
            rows = [r for r in results if r[0] == name and r[1] == scen]
            falls = sum(r[3] for r in rows)
            mean_t = sum(r[4] for r in rows) / len(rows)
            print(f"TANI2 {name} | {scen} | düşen={falls}/{len(rows)} | ort. süre={mean_t:.2f} s (maks ~14.75 s)", flush=True)


if __name__ == "__main__":
    main()