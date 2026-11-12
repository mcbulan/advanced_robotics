"""Create the allocation strategies (alpha vectors) and one fine-tuning config per strategy.
Budget rule: sum_g alpha_g*D_g = rho * sum_g D_g (same nominal assistance-moment budget for every strategy),
alpha_g = rho*sumD*w_g/D_g, capped at ALPHA_MAX. Realised budget after capping is printed and saved.
Strategies: none | hip | knee | ankle | uniform (budget split 1/3 each) | demand (budget split in proportion to D_g = equal ratio).
Usage (repo root): python scripts/make_strategy_configs.py --demand demand.json --rho 0.3 --warm_start <model.zip> --payload_kg 20 --outdir configs"""
import argparse, json, os, subprocess, sys
ALPHA_MAX = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--demand", required=True); ap.add_argument("--rho", type=float, default=0.3)
ap.add_argument("--warm_start", required=True, help="trained model .zip of the unassisted policy (same payload as below)")
ap.add_argument("--payload_kg", type=float, default=20.0); ap.add_argument("--payload_mode", default="back")
ap.add_argument("--umax", type=float, default=60.0, help="peak torque limit per joint [Nm] (ASSUMED)")
ap.add_argument("--timesteps", type=int, default=3_000_000); ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--n_steps", type=int, default=None); ap.add_argument("--batch", type=int, default=None); ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--outdir", default="configs"); ap.add_argument("--src", default=None); ap.add_argument("--slope_deg", type=float, default=None, help="uphill slope [deg]; omit = flat")
a = ap.parse_args()
D = json.load(open(a.demand)); Dg = dict(hip=D["D_hip"], knee=D["D_knee"], ankle=D["D_ankle"]); S = sum(Dg.values())
W = dict(none=None, hip=dict(hip=1, knee=0, ankle=0), knee=dict(hip=0, knee=1, ankle=0), ankle=dict(hip=0, knee=0, ankle=1),
         uniform=dict(hip=1/3, knee=1/3, ankle=1/3), demand={g: Dg[g] / S for g in Dg})
os.makedirs(a.outdir, exist_ok=True); out = {}
for name, w in W.items():
    al = dict(hip=0.0, knee=0.0, ankle=0.0)
    if w:
        for g in Dg: al[g] = min(ALPHA_MAX, a.rho * S * w[g] / Dg[g])
    real = sum(al[g] * Dg[g] for g in Dg) / S
    out[name] = dict(alpha=al, umax=dict(hip=a.umax, knee=a.umax, ankle=a.umax), budget_rho_target=a.rho if w else 0.0, budget_rho_realised=real)
    assist = json.dumps(dict(alpha=al, umax=out[name]["umax"]))
    tag = f"_s{int(a.slope_deg)}" if a.slope_deg else ""
    cfgp = os.path.join(a.outdir, f"ft_{name}_p{int(a.payload_kg)}{tag}.json")
    cmd = [sys.executable, os.path.join(HERE, "make_configs.py"), "--msk", "myolegs22", "--out", cfgp, "--payload_kg", str(a.payload_kg),
           "--payload_mode", a.payload_mode, "--num_envs", str(a.num_envs), "--timesteps", str(a.timesteps), "--assist_json", assist,
           "--warm_start", a.warm_start, "--lr", str(a.lr)]
    if a.slope_deg: cmd += ["--slope_deg", str(a.slope_deg)]
    if a.src: cmd += ["--src", a.src]
    for k, v in (("--n_steps", a.n_steps), ("--batch", a.batch), ("--epochs", a.epochs)):
        if v is not None: cmd += [k, str(v)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"{name:8s} alpha(h,k,a)=({al['hip']:.2f},{al['knee']:.2f},{al['ankle']:.2f})  realised budget={real:.2f}  -> {cfgp}")
json.dump(dict(demand=Dg, rho=a.rho, payload_kg=a.payload_kg, strategies=out), open(os.path.join(a.outdir, "strategies.json"), "w"), indent=1)
print("saved", os.path.join(a.outdir, "strategies.json"))
