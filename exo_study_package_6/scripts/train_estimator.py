"""Train the load estimator (small MLP 19-32-32-1, tanh) on est_data.npz; split BY ROLLOUT (no leakage between windows of one episode).
Prints validation RMSE/MAE in kg and two reference baselines (constant = training mean; linear ridge). Saves numpy weights (est.npz).
Usage: python scripts/train_estimator.py --data est_data.npz --out est.npz"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch_files"))
from learned_controller import save_estimator
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="est.npz")
ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
z = np.load(a.data); X, y, g = z["X"], z["y"], z["g"]; rng = np.random.default_rng(a.seed)
ids = np.unique(g); rng.shuffle(ids); val_ids = set(ids[: max(1, len(ids) // 5)].tolist()); isv = np.array([i in val_ids for i in g])
mu, sd = X[~isv].mean(0), X[~isv].std(0) + 1e-6; Xn = (X - mu) / sd
torch.manual_seed(a.seed); net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
Xt, yt = torch.tensor(Xn[~isv], dtype=torch.float32), torch.tensor(y[~isv], dtype=torch.float32)[:, None]
Xv, yv = torch.tensor(Xn[isv], dtype=torch.float32), torch.tensor(y[isv], dtype=torch.float32)[:, None]
best, state = 1e9, None
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]; opt.zero_grad(); torch.nn.functional.mse_loss(net(Xt[b]), yt[b]).backward(); opt.step()
    with torch.no_grad(): v = torch.nn.functional.mse_loss(net(Xv), yv).item()
    if v < best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
net.load_state_dict(state)
with torch.no_grad(): pv = net(Xv).numpy().ravel()
yv_ = y[isv]; A = np.c_[Xn[~isv], np.ones((~isv).sum())]; w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[~isv])
plin = np.c_[Xn[isv], np.ones(isv.sum())] @ w
rm = lambda p: float(np.sqrt(np.mean((p - yv_) ** 2)))
print(f"validation windows: {isv.sum()} (from {len(val_ids)} rollouts)\n  constant baseline RMSE = {rm(np.full_like(yv_, y[~isv].mean())):.2f} kg\n  ridge RMSE            = {rm(plin):.2f} kg\n  MLP RMSE / MAE        = {rm(pv):.2f} / {np.mean(np.abs(pv - yv_)):.2f} kg")
save_estimator(a.out, net, mu, sd); print("saved", a.out, "(report RMSE in the paper; this is the sensing error the controller lives with)")
