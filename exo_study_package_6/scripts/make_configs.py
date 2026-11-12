"""Config generator (study code). Run from the MyoAssist repo root. Derives MyoAssist imitation configs from the shipped Hippo_L1 myolegs22 config.
  legs22 : myolegs22 + Hippo_L1 (+ optional payload / population)      -> indexing unchanged (44 obs, 22 muscles)
  fullbody: myofullbody + Hippo_L1: indexing REWRITTEN for n_muscle muscles (read from the composed model)
UNVERIFIED for learning: only environment creation + a tiny training run (pipeline/shape test) are checked."""
import json, copy, argparse, os
import mujoco
from myoassist_utils.env_spec import EnvSpec
SRC = "rl_train/train/train_configs/device_sweep/imitation_22_Hippo_L1_h128_e32_sidenet_mirror0p1_actpen10.json"  # relative to the repo root

def model_dims(msk):
    xml = EnvSpec(msk=msk, device="Hippo_L1", terrain=None).validate().compose(planar_root=True)
    m = mujoco.MjModel.from_xml_string(xml)
    nm = int((m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).sum())
    assert (m.actuator_dyntype[:nm] == mujoco.mjtDyn.mjDYN_MUSCLE).all() and m.nu == nm + 2
    return nm

def rewrite_fullbody(c, nm):
    N_Q, N_V, N_S = 8, 9, 4
    a0, a1 = N_Q + N_V, N_Q + N_V + nm          # muscle activations
    s0, s1 = a1, a1 + N_S                        # foot sensors
    tv = s1                                      # target velocity
    obs = [dict(type="range", range=r, comment=t) for r, t in
           (([0, N_Q], "qpos"), ([N_Q, a0], "qvel"), ([a0, a1], f"{nm} muscle activations"), ([s0, s1], "4 foot force"), ([tv, tv + 1], "target velocity"))]
    pi = c["policy_params"]["custom_policy_params"]
    ni = pi["net_indexing_info"]
    ni["human_actor"]["observation"] = copy.deepcopy(obs)
    ni["common_critic"]["observation"] = copy.deepcopy(obs)
    ni["human_actor"]["action"] = [dict(type="range_mapping", range_net=[0, nm], range_action=[0, nm], comment=f"{nm} muscles, no exo")]
    sh = s0 - 39                                  # sensor offset relative to the 22-muscle layout
    for k, (lo, hi) in (("exo_actor_r", (nm, nm + 1)), ("exo_actor_l", (nm + 1, nm + 2))):
        idx = ni[k]["observation"][0]["index"]
        ni[k]["observation"][0]["index"] = [i + sh if i >= 39 else i for i in idx]
        ni[k]["action"][0]["range_action"] = [lo, hi]
    c["env_params"].update(msk_key="myofullbody", planar_root=True)
    # myofullbody has no r/l_mtp_sensor (checked in the composed model): drop them from the joint-limit penalty
    c["env_params"]["joint_limit_sensor_keys"] = ["r_knee_sensor", "l_knee_sensor", "r_hip_sensor", "l_hip_sensor", "r_ankle_sensor", "l_ankle_sensor"]
    c["ppo_params"]["mirror_coef"] = 0.0         # mirror permutation is name-based for 22 muscles: disabled, NOT validated for 416
    pi["net_arch"]["human_actor"] = [256, 256]    # wider actor for 416 outputs (ASSUMED, untuned)
    return c

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--msk", choices=["myolegs22", "myofullbody"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--payload_kg", type=float, default=0.0)
    ap.add_argument("--payload_mode", default="back")
    ap.add_argument("--num_envs", type=int, default=12)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n_steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--src", default=SRC, help="source Hippo_L1 imitation config (default: shipped 22-muscle Hippo_L1 config)")
    ap.add_argument("--assist_json", default=None, help="JSON string for env_params.assist (assist overlay)")
    ap.add_argument("--warm_start", default=None, help="path of a trained model .zip to continue from (prev_trained_policy_path)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--slope_deg", type=float, default=None, help="uphill slope in degrees (walking direction +x); omitted = flat ground")
    a = ap.parse_args()
    c = json.load(open(a.src))
    if a.msk == "myofullbody":
        c = rewrite_fullbody(c, model_dims("myofullbody"))
    e = c["env_params"]
    e.update(num_envs=a.num_envs, payload_kg=a.payload_kg, payload_mode=a.payload_mode)
    if a.assist_json: e["assist"] = json.loads(a.assist_json)
    if a.warm_start: e["prev_trained_policy_path"] = a.warm_start
    if a.lr is not None: c["ppo_params"]["learning_rate"] = a.lr
    if a.slope_deg:
        tp = os.path.splitext(a.out)[0] + "_terrain.json"
        json.dump({"terrain": "slope", "deg": a.slope_deg}, open(tp, "w")); e["terrain"] = tp   # path is relative to the repo root (run from there)
    c["total_timesteps"] = a.timesteps
    for k, v in (("n_steps", a.n_steps), ("batch_size", a.batch), ("n_epochs", a.epochs)):
        if v is not None: c["ppo_params"][k] = v
    json.dump(c, open(a.out, "w"), indent=1)
    print("written", a.out)
