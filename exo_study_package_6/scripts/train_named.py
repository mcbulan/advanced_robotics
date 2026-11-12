"""Run rl_train/run_train.py for a config and register the resulting session directory under a name in sessions.json.
Usage (repo root, PYTHONPATH=.):  python scripts/train_named.py --name base_p0 --config configs/base_p0.json [--registry sessions.json]"""
import argparse, json, os, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--registry", default="sessions.json"); a = ap.parse_args()
reg = json.load(open(a.registry)) if os.path.exists(a.registry) else {}
p = subprocess.Popen([sys.executable, "rl_train/run_train.py", "--config_file_path", a.config], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
sess = None
for line in p.stdout:
    print(line, end="")
    if line.startswith("Session directory:"):
        sess = line.split(":", 1)[1].strip(); reg[a.name] = sess; json.dump(reg, open(a.registry, "w"), indent=1)
p.wait()
print(f"\n[train_named] {a.name} -> {sess} (exit {p.returncode}); registry: {a.registry}")
