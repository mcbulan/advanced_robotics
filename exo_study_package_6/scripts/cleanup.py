"""Delete old training sessions to free disk space. DRY RUN by default (only lists sizes).
Usage (repo root): python exo_study_package/scripts/cleanup.py --keep base_p20 base_p20_c1 [--registry work/sessions.json] [--delete]
Keeps the sessions named in --keep (names from sessions.json) and the registry file itself; removes all other rl_train/results/train_session_* folders."""
import argparse, glob, json, os, shutil
ap = argparse.ArgumentParser(); ap.add_argument("--keep", nargs="+", required=True)
ap.add_argument("--registry", default="work/sessions.json"); ap.add_argument("--delete", action="store_true"); a = ap.parse_args()
reg = json.load(open(a.registry)) if os.path.exists(a.registry) else {}
missing = [k for k in a.keep if k not in reg]
if missing: raise SystemExit(f"names not in {a.registry}: {missing}  (registered: {list(reg)})")
keep = {os.path.normpath(reg[k]) for k in a.keep}
def size(d): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6
tot = 0
for d in sorted(glob.glob("rl_train/results/train_session_*")):
    k = os.path.normpath(d) in keep; mb = size(d)
    print(f"{'KEEP  ' if k else 'DELETE'} {mb:9.1f} MB  {d}")
    if not k:
        tot += mb
        if a.delete: shutil.rmtree(d)
print(f"\n{'deleted' if a.delete else 'would delete'}: {tot:.1f} MB" + ("" if a.delete else "   (dry run; add --delete to remove)"))
