"""Shared helpers (study code). Run scripts from the MyoAssist repo root with PYTHONPATH=."""
import copy, glob, json, os, re
import numpy as np

JOINTS = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]

def latest_model(session):
    zs = glob.glob(os.path.join(session, "trained_models", "model_*.zip"))
    if not zs: raise SystemExit("no model_*.zip under " + session)
    return max(zs, key=lambda p: int(re.findall(r"model_(\d+)", p)[-1]))

def load_policy(session, model_path=None):
    import stable_baselines3
    from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy
    from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy
    cfg0 = json.load(open(os.path.join(session, "session_config.json")))
    exo = cfg0["env_id"].endswith("Exo-v0") if "env_id" in cfg0 else cfg0["env_params"]["env_id"].endswith("Exo-v0")
    pol = HumanExoActorCriticPolicy if exo else HumanActorCriticPolicy
    model = stable_baselines3.PPO.load(model_path or latest_model(session), custom_objects={"policy_class": pol})
    return model, cfg0

def make_env(cfg0, *, payload_kg=0.0, payload_mode="back", population=None, pop_seed=0, assist=None, keep_ref_termination=False):
    """Create ONE evaluation environment from a session_config dict with the given overrides."""
    from rl_train.utils.data_types import DictionableDataclass
    from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
    from rl_train.envs.environment_handler import EnvironmentHandler
    cfg = DictionableDataclass.create(ImitationTrainSessionConfig, copy.deepcopy(cfg0))
    e = cfg.env_params
    e.num_envs = 1; e.payload_kg = payload_kg; e.payload_mode = payload_mode
    e.population_params = population; e.population_seed = pop_seed; e.assist = assist
    if not keep_ref_termination: e.out_of_trajectory_threshold = 1e9   # fall = pelvis height < safe_height only
    return EnvironmentHandler.create_environment(cfg, False, is_evaluate_mode=True)

def rollout(env, model, steps, speed, record=True, hook=None, return_trace=False):
    """Roll out one deterministic episode. Returns metrics dict. Works with or without an assist overlay."""
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
    m = env.sim.model.ptr if hasattr(env.sim.model, "ptr") else env.sim.model
    env.set_target_velocity_mode_manually(mode=MyoAssistLegBase.VelocityMode.UNIFORM, starting_phase=0,
        initial_target_velocity=speed, min_target_velocity=speed, max_target_velocity=speed, target_velocity_period=2)
    obs, _ = env.reset()
    dev = np.asarray(env._device_actuator_ids, dtype=int)
    mus = np.setdiff1d(np.arange(m.nu), dev)
    dofs = [m.jnt_dofadr[m.joint(n).id] for n in JOINTS]
    mass = float(m.body_mass.sum()); weight = mass * 9.81
    ov = getattr(env, "_assist", None)
    x0 = float(env.sim.data.joint("pelvis_tx").qpos[0]); n = 0; fell = False
    acc = {k: [] for k in ("E", "B", "V", "P", "TB", "TE", "RATE")}
    trace = []
    for i in range(steps):
        if hook is not None: hook(i, env)
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(np.array(a, dtype=float))
        d = env.sim.data
        acc["E"].append(float(np.mean(d.act[mus] ** 2)))
        acc["B"].append(np.abs(d.qfrc_actuator[dofs]))                 # |net muscle moment|, 6 joints
        weight = float(m.body_mass.sum()) * 9.81                         # current weight (changes in a load step)
        acc["V"].append(float(env._get_max_joint_constraint_force() / weight))
        if ov is not None:
            acc["P"].append(ov.power); acc["TE"].append(np.abs(ov.tau_exo)); acc["RATE"].append(ov.rate)
        if return_trace:
            trace.append(dict(t=(i + 1) * env.dt, Rbio=float(np.abs(d.qfrc_actuator[dofs]).mean() / float(m.body_mass.sum())),
                              tauBio=float(np.abs(d.qfrc_actuator[dofs]).mean()), P=float(ov.power) if ov is not None else 0.0,
                              L_hat=float(ov.lc.L_hat) if (ov is not None and ov.lc is not None) else float("nan"),
                              mass=float(m.body_mass.sum()), vc=float(acc["V"][-1])))
        n += 1
        if term: fell = True; break
        if trunc: break
    x1 = float(env.sim.data.joint("pelvis_tx").qpos[0]); T = n * env.dt
    B = np.mean(acc["B"], axis=0)                                       # per-joint mean |tau_bio| (Nm)
    out = dict(survived_s=T, fell=int(fell), distance_m=x1 - x0, mean_speed=(x1 - x0) / T if T > 0 else 0.0,
               E_act=float(np.mean(acc["E"])), R_bio=float(B.mean() / mass), V_c=float(np.mean(acc["V"])),
               P_exo=float(np.mean(acc["P"])) if acc["P"] else 0.0)
    if acc["P"]:
        out.update(P_p95=float(np.percentile(acc["P"], 95)), P_max=float(np.max(acc["P"])), rate_p99=float(np.percentile(acc["RATE"], 99)))
    out["V_c_p95"] = float(np.percentile(acc["V"], 95)) if acc["V"] else 0.0
    if ov is not None and ov.nstep:
        out.update(viol_rate=ov.nviol["rate"] / ov.nstep, viol_power=ov.nviol["power"] / ov.nstep, viol_vc=ov.nviol["vc"] / ov.nstep)
        if ov.lc is not None: out["L_hat_end"] = float(ov.lc.L_hat)
    for j, nme in enumerate(JOINTS): out["tauBio_" + nme] = float(B[j])
    if acc["TE"]:
        TE = np.mean(acc["TE"], axis=0)
        for j, nme in enumerate(JOINTS): out["tauExo_" + nme] = float(TE[j])
    return (out, trace) if return_trace else out
