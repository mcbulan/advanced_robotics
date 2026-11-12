"""Create the fine-tuning config of the learned controller (policy trained WITH the learned_qp controller in the loop).
Usage (repo root): python scripts/make_learned_config.py --estimator est.npz --limits limits.json --demand demand.json --alloc alloc.json(optional) \
   --warm_start <ft_none model.zip> --payload_kg 20 [--slope_deg 8] --out work/ft_learned_p20.json
Absolute paths are written into the config (estimator_path)."""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--estimator", required=True); ap.add_argument("--limits", required=True); ap.add_argument("--demand", required=True)
ap.add_argument("--alloc", default=None); ap.add_argument("--warm_start", required=True); ap.add_argument("--payload_kg", type=float, default=20.0)
ap.add_argument("--payload_mode", default="back"); ap.add_argument("--umax", type=float, default=60.0); ap.add_argument("--timesteps", type=int, default=3_000_000)
ap.add_argument("--num_envs", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--slope_deg", type=float, default=None)
ap.add_argument("--src", default=None); ap.add_argument("--out", required=True); a = ap.parse_args()
D = json.load(open(a.demand)); lim = json.load(open(a.limits)); lim.pop("note", None)
lc = dict(estimator_path=os.path.abspath(a.estimator), d_nominal=dict(hip=D["D_hip"], knee=D["D_knee"], ankle=D["D_ankle"]))
if a.alloc: lc["alloc"] = json.load(open(a.alloc))
assist = dict(controller="learned_qp", alpha=dict(hip=0, knee=0, ankle=0), umax=dict(hip=a.umax, knee=a.umax, ankle=a.umax), limits=lim, lc=lc)
cmd = [sys.executable, os.path.join(HERE, "make_configs.py"), "--msk", "myolegs22", "--out", a.out, "--payload_kg", str(a.payload_kg), "--payload_mode", a.payload_mode,
       "--num_envs", str(a.num_envs), "--timesteps", str(a.timesteps), "--assist_json", json.dumps(assist), "--warm_start", a.warm_start, "--lr", str(a.lr)]
if a.slope_deg: cmd += ["--slope_deg", str(a.slope_deg)]
if a.src: cmd += ["--src", a.src]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL); json.dump(assist, open(os.path.splitext(a.out)[0] + "_assist.json", "w"), indent=1)
print("saved", a.out, "and", os.path.splitext(a.out)[0] + "_assist.json")
