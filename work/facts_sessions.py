import glob, json, os, datetime as dt
def t(s): return dt.datetime.strptime(s.split(".")[0], "%Y%m%d-%H%M%S")
for d in sorted(glob.glob("rl_train/results/train_session_*")):
    fs = glob.glob(d + "/**/train_log.json", recursive=True)
    if not fs: print(os.path.basename(d), "train_log yok"); continue
    L = json.load(open(fs[0]))["log_datas"]
    a, b = t(L[0]["time"]), t(L[-1]["time"])
    print(os.path.basename(d)[14:], "adim=%d" % L[-1]["num_timesteps"], "sure_saat=%.1f" % ((b - a).total_seconds() / 3600), "son=" + L[-1]["time"][:15])
