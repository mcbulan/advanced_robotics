"""Install the study patch into a MyoAssist checkout (idempotent, text-anchored, version tolerant).
Usage:  python scripts/install_patch.py --repo /path/to/myoassist
Does: copies env_perturbations.py + assist_overlay.py + learned_controller.py into rl_train/envs, adds EnvParams fields (payload/population/assist),
      adds 2 hooks in MyoAssistLegBase (_setup, step). Backups: *.orig_study (made once)."""
import argparse, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); PF = os.path.join(HERE, "..", "patch_files")
FIELDS = '''
        # STUDY_PATCH_FIELDS: carried load, virtual-population parameters, assist overlay (defaults = unchanged behaviour)
        payload_kg: float = 0.0
        payload_mode: str = "back"          # "back" | "front"
        payload_body: str = "torso"
        population_params: dict = None
        population_seed: int = 0
        assist: dict = None                 # see rl_train/envs/assist_overlay.py
'''
SETUP = '''
        # STUDY_PATCH_SETUP
        from rl_train.envs import env_perturbations as _study_ep
        _study_ep.setup_all(self, env_params)
'''
STEP = '''        # STUDY_PATCH_STEP
        from rl_train.envs.assist_overlay import assist_pre_step as _study_pre
        a = _study_pre(self, a)
'''

def backup(p):
    if not os.path.exists(p + ".orig_study"): shutil.copyfile(p, p + ".orig_study")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); r = ap.parse_args().repo
    envs = os.path.join(r, "rl_train", "envs"); cfgp = os.path.join(r, "rl_train", "train", "train_configs", "config.py")
    basep = os.path.join(envs, "myoassist_leg_base.py")
    for p in (envs, cfgp, basep):
        if not os.path.exists(p): sys.exit(f"not found: {p}  (is --repo the MyoAssist root?)")
    for f in ("env_perturbations.py", "assist_overlay.py", "learned_controller.py"): shutil.copyfile(os.path.join(PF, f), os.path.join(envs, f))
    # 1) EnvParams
    t = open(cfgp).read()
    if "STUDY_PATCH_FIELDS" not in t:
        m = re.search(r"^(\s+)device_ctrl_scale\s*:\s*float\s*=\s*[\d.]+\s*$", t, re.M) or re.search(r"^(\s+)planar_root\s*:\s*bool\s*=\s*\w+\s*$", t, re.M)
        if not m: sys.exit("config.py: anchor (device_ctrl_scale / planar_root) not found. Add the fields in FIELDS to EnvParams by hand.")
        backup(cfgp); t = t[:m.end()] + "\n" + FIELDS + t[m.end():]; open(cfgp, "w").write(t); print("config.py patched")
    else: print("config.py already patched")
    # 2) env base class hooks
    b = open(basep).read(); changed = False
    if "STUDY_PATCH_SETUP" not in b:
        m = re.search(r"^[ \t]+self\.sim\.model\.opt\.timestep\s*=.*$", b, re.M)
        if not m: sys.exit("myoassist_leg_base.py: timestep anchor not found in _setup. Insert SETUP block by hand after model setup.")
        b = b[:m.end()] + "\n" + SETUP.rstrip("\n") + b[m.end():]; changed = True
    if "STUDY_PATCH_STEP" not in b:
        m = re.search(r"def step\(self, a, \*\*kwargs\):\n(\s+)self\._modulate_target_velocity\(\)\n", b)
        if not m: sys.exit("myoassist_leg_base.py: step() anchor not found. Insert STEP block at the start of MyoAssistLegBase.step by hand.")
        b = b[:m.end()] + STEP + b[m.end():]; changed = True
    if changed: backup(basep); open(basep, "w").write(b); print("myoassist_leg_base.py patched")
    else: print("myoassist_leg_base.py already patched")
    print("OK. Next: python scripts/test_assist.py  (from the repo root, PYTHONPATH=.)")

if __name__ == "__main__": main()
