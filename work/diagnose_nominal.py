"""Run from the MyoAssist repository root; preserve each policy's assist overlay."""
import copy
import json
import os
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.abspath(os.environ.get("P", "exo_study_package_6/scripts")))
from common import load_policy, make_env, rollout

with open("work/sessions.json", encoding="utf-8") as handle:
    registry = json.load(handle)
with open("work/cfg/strategies.json", encoding="utf-8") as handle:
    strategies = json.load(handle)["strategies"]
with open("work/ft_learned_p20_assist.json", encoding="utf-8") as handle:
    learned_assist = json.load(handle)

cases = [
    ("none", "ft_none_p20", strategies["none"]),
    ("demand", "ft_demand_p20", strategies["demand"]),
    ("learned_qp", "ft_learned_p20", learned_assist),
]

for name, key, original_assist in cases:
    assist = copy.deepcopy(original_assist)
    assist.pop("budget_rho_target", None)
    assist.pop("budget_rho_realised", None)
    model, config = load_policy(registry[key])
    env = make_env(config, payload_kg=20.0, population=None, assist=assist)
    try:
        result = rollout(env, model, steps=600, speed=1.25)
    finally:
        env.close()
    passed = (
        not result["fell"]
        and result["survived_s"] >= 10.0
        and abs(result["mean_speed"] - 1.25) <= 0.35
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"TANI {name}: yük=20 kg, nominal; "
        f"süre={result['survived_s']:.2f} s; "
        f"düşme={result['fell']}; "
        f"hız={result['mean_speed']:.3f} m/s; {status}",
        flush=True,
    )