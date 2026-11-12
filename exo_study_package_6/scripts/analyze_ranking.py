"""Ranking stability analysis (pure pandas/scipy; no simulator needed). Input: results.csv from eval_strategies.py.
Primary outcome (lower = better): --metric (default E_act = mean squared muscle activation, an activation PROXY, not metabolic energy).
A fallen rollout (fell=1) is ranked after all non-fallen ones (among fallen: longer survival first). For each payload:
  - per-individual strategy ranking; mean rank; frequency of rank 1
  - rank stability: Kendall tau_b between the nominal ranking and each individual's ranking (mean, min)  ; mean pairwise tau_b among individuals
  - paired difference vs 'none' (Delta = metric_strategy - metric_none) : mean, bootstrap 95% CI, Wilcoxon signed-rank p (Holm-corrected)
Usage: python scripts/analyze_ranking.py --csv results.csv --metric E_act --out summary.md"""
import argparse, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import kendalltau, wilcoxon
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--metric", default="E_act")
ap.add_argument("--out", default="summary.md"); ap.add_argument("--n_boot", type=int, default=5000); a = ap.parse_args()
df = pd.read_csv(a.csv); rng = np.random.default_rng(0); L = []
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
for pk, g in df.groupby("payload_kg"):
    L.append(f"\n## payload {pk:g} kg\n")
    g = g.copy(); g["score"] = np.where(g["fell"] == 1, 1e6 - g["survived_s"], g[a.metric])   # non-fallen first (by metric); fallen after, longer survival first
    piv = g.pivot_table(index="individual", columns="strategy", values="score", aggfunc="first")
    fall = g.pivot_table(index="individual", columns="strategy", values="fell", aggfunc="first")
    rk = piv.rank(axis=1, method="average")
    L.append("Falls (count / n):  " + ", ".join(f"{c}: {int(fall[c].sum())}/{len(fall)}" for c in fall.columns) + "\n")
    L.append("| strategy | mean rank | rank-1 share | median " + a.metric + " (non-fallen) |\n|---|---|---|---|")
    first = (rk == rk.min(axis=1).values[:, None]).mean()
    for c in rk.columns:
        med = g[(g.strategy == c) & (g.fell == 0)][a.metric].median()
        L.append(f"| {c} | {rk[c].mean():.2f} | {first[c]:.2f} | {med:.4g} |")
    pop = [i for i in rk.index if str(i) != "nominal"]
    if "nominal" in rk.index and len(pop) > 1:
        taus = [kendalltau(rk.loc["nominal"], rk.loc[i], variant="b")[0] for i in pop]
        pw = [kendalltau(rk.loc[i], rk.loc[j], variant="b")[0] for i, j in itertools.combinations(pop, 2)]
        L.append(f"\nKendall tau_b nominal vs individuals: mean {np.nanmean(taus):.2f}, min {np.nanmin(taus):.2f}; pairwise among individuals: mean {np.nanmean(pw):.2f}  (NaN = ties only)")
    if "none" in piv.columns:
        L.append("\n| strategy vs none | mean delta | 95% CI (bootstrap) | n | Wilcoxon p (Holm) |\n|---|---|---|---|---|")
        ok = piv.loc[pop] if pop else piv; res = []
        for c in [c for c in piv.columns if c != "none"]:
            okf = fall.loc[ok.index]; d = (ok[c] - ok["none"])[(okf[c] == 0) & (okf["none"] == 0)].dropna().values
            if len(d) < 5: res.append((c, np.nan, np.nan, np.nan, len(d), np.nan)); continue
            bs = [rng.choice(d, len(d)).mean() for _ in range(a.n_boot)]
            try: pv = wilcoxon(d).pvalue
            except ValueError: pv = np.nan
            res.append((c, d.mean(), *np.percentile(bs, [2.5, 97.5]), len(d), pv))
        ps = holm(np.array([r[5] if not np.isnan(r[5]) else 1.0 for r in res]))
        for r, pa in zip(res, ps): L.append(f"| {r[0]} | {r[1]:.4g} | [{r[2]:.4g}, {r[3]:.4g}] | {r[4]} | {pa:.3g} |")
        L.append("\n(delta computed only over individuals where both rollouts did not fall.)")
open(a.out, "w").write("\n".join(L)); print("\n".join(L))
