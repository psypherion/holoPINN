# ==============================================================
# MLP + Sobolev ablation — single run
# Tests whether Sobolev supervision alone (without holonomy)
# achieves the accuracy of holo-Diff+Sob.
# If this gives W1,1 ~ 0.0216 (same as MLP-Only) rather than
# 0.0018 (holo-Diff+Sob), holonomy is causally necessary.
# ==============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("results", exist_ok=True)

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

# ── Config (identical to main experiment) ─────────────────────
class Config:
    nu             = 0.01 / np.pi
    seed           = 42
    hidden         = 128
    n_layers       = 4
    n_fourier      = 64
    fourier_sigma  = 1.0        # best σ from sweep
    epochs_adam    = 20_000
    lr             = 1e-3
    lr_min         = 1e-5
    epochs_lbfgs   = 5_000
    lbfgs_lr       = 1.0
    lbfgs_max_iter = 20
    n_colloc       = 512
    batch_size     = 2048
    log_every      = 1000
    lra_enabled      = True
    lra_update_every = 1000
    lra_alpha        = 0.9
    lra_lambda_max   = 100.0
    phys_weight_init = 1.0
    sob_weight     = 0.05       # same as holo-Diff+Sob

# ── Fourier MLP (pure MLP, no GNN) ────────────────────────────
class FourierMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        B = torch.randn(2, cfg.n_fourier) * cfg.fourier_sigma
        self.register_buffer("B", B)
        in_dim = 2 * cfg.n_fourier
        layers = [nn.Linear(in_dim, cfg.hidden), nn.Tanh()]
        for _ in range(cfg.n_layers - 2):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh()]
        layers.append(nn.Linear(cfg.hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        proj = xt @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat)

# ── LRA ───────────────────────────────────────────────────────
def lra_update(model, loss_data, loss_pde, lambda_p, alpha, lambda_max):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return lambda_p
    try:
        gd = torch.autograd.grad(loss_data, trainable,
                                  retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(loss_pde,  trainable,
                                  retain_graph=True, allow_unused=True)
    except RuntimeError:
        return lambda_p
    nd = sum(g.norm().item()**2 for g in gd if g is not None)**0.5
    np_ = sum(g.norm().item()**2 for g in gp if g is not None)**0.5
    if np_ > 1e-10:
        lambda_p = alpha * lambda_p + (1 - alpha) * (nd / np_)
        lambda_p = min(lambda_p, lambda_max)
    return float(lambda_p)

# ── Burgers residual (on MLP only) ────────────────────────────
def burgers_residual(model, xt_sub, nu):
    u    = model(xt_sub)
    g    = torch.autograd.grad(u, xt_sub,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x  = g[:, 0:1]; u_t = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_sub,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    return torch.mean((u_t + u * u_x - nu * u_xx) ** 2)

# ── Metrics ───────────────────────────────────────────────────
def rel_l2(pred, exact):
    return float(np.linalg.norm(pred - exact) /
                 (np.linalg.norm(exact) + 1e-12))

def w1_1_rel_error(u_pred, exact, x_vec, t_vec):
    nt, nx = exact.shape
    u2d = u_pred.reshape(nt, nx)
    dx  = float(x_vec[1] - x_vec[0])
    dt  = float(t_vec[1] - t_vec[0])
    num = (np.mean(np.abs(u2d - exact)) +
           np.mean(np.abs(np.gradient(u2d,   dx, axis=1) -
                          np.gradient(exact, dx, axis=1))) +
           np.mean(np.abs(np.gradient(u2d,   dt, axis=0) -
                          np.gradient(exact, dt, axis=0))))
    den = (np.mean(np.abs(exact)) +
           np.mean(np.abs(np.gradient(exact, dx, axis=1))) +
           np.mean(np.abs(np.gradient(exact, dt, axis=0))))
    return float(num / (den + 1e-12))

# ── Data ──────────────────────────────────────────────────────
def load_data(path="../datasets/Burgers.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    data  = np.load(path)
    t_vec = data["t"].squeeze()
    x_vec = data["x"].squeeze()
    exact = data["usol"].T          # (Nt, Nx)
    xx, tt = np.meshgrid(x_vec, t_vec)
    X = np.vstack([xx.ravel(), tt.ravel()]).T
    y = exact.reshape(-1, 1)
    print(f"[Data] t∈[{t_vec.min():.3f},{t_vec.max():.3f}]  "
          f"x∈[{x_vec.min():.3f},{x_vec.max():.3f}]  "
          f"Nt={len(t_vec)} Nx={len(x_vec)} N={X.shape[0]}")
    return (X.astype(np.float32), y.astype(np.float32),
            t_vec, x_vec, exact)

# ── Training ──────────────────────────────────────────────────
def train(cfg, X_np, y_np, t_vec, x_vec, exact, device):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Nt, Nx = len(t_vec), len(x_vec)
    N      = X_np.shape[0]
    dx_val = float(x_vec[1] - x_vec[0])
    dt_val = float(t_vec[1] - t_vec[0])

    # precompute Sobolev targets
    du_x = np.gradient(exact, dx_val, axis=1).astype(np.float32)
    du_t = np.gradient(exact, dt_val, axis=0).astype(np.float32)
    du_x_dev = torch.from_numpy(du_x).to(device)
    du_t_dev = torch.from_numpy(du_t).to(device)

    X_dev = torch.from_numpy(X_np).to(device)
    y_dev = torch.from_numpy(y_np).to(device)

    model     = FourierMLP(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs_adam, eta_min=cfg.lr_min)
    lambda_p  = cfg.phys_weight_init

    history = {"data": [], "phys": [], "sob": [], "epoch": [], "lambda_p": []}

    print(f"\n{'─'*64}")
    print(f"  Run: MLP+Sobolev  |  σ={cfg.fourier_sigma}")
    print(f"  hidden={cfg.hidden} layers={cfg.n_layers} "
          f"Adam={cfg.epochs_adam} L-BFGS={cfg.epochs_lbfgs}")
    print(f"{'─'*64}")

    # ── Adam ──────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs_adam + 1):
        model.train()
        optimizer.zero_grad()

        u_all     = model(X_dev)
        idx_b     = torch.randint(0, N, (cfg.batch_size,), device=device)
        data_loss = F.mse_loss(u_all[idx_b], y_dev[idx_b])

        idx_c     = np.random.choice(N, cfg.n_colloc, replace=False)
        xt_c      = torch.from_numpy(X_np[idx_c]).to(device).requires_grad_(True)
        phys_loss = burgers_residual(model, xt_c, cfg.nu)

        if cfg.lra_enabled and epoch % cfg.lra_update_every == 0 and epoch > 1:
            lambda_p = lra_update(model, data_loss, phys_loss,
                                  lambda_p, cfg.lra_alpha, cfg.lra_lambda_max)

        # Sobolev FD loss on full grid
        u_grid = u_all.reshape(Nt, Nx)
        dpx    = (u_grid[:, 2:] - u_grid[:, :-2]) / (2.0 * dx_val)
        dpt    = (u_grid[2:, :] - u_grid[:-2, :]) / (2.0 * dt_val)
        sob_loss = 0.5 * (
            F.l1_loss(dpx, du_x_dev[:, 1:-1]) /
                (du_x_dev[:, 1:-1].abs().mean() + 1e-8) +
            F.l1_loss(dpt, du_t_dev[1:-1, :]) /
                (du_t_dev[1:-1, :].abs().mean() + 1e-8)
        )

        loss = data_loss + lambda_p * phys_loss + cfg.sob_weight * sob_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["data"].append(data_loss.item())
        history["phys"].append(phys_loss.item())
        history["sob"].append(sob_loss.item())
        history["epoch"].append(epoch)
        history["lambda_p"].append(lambda_p)

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:6d} | Data {data_loss.item():.3e} "
                  f"| PDE {phys_loss.item():.3e} "
                  f"| Sob {sob_loss.item():.2e} "
                  f"| λ_p {lambda_p:.2f} "
                  f"| LR {scheduler.get_last_lr()[0]:.2e}")

    # ── L-BFGS ────────────────────────────────────────────────
    print(f"\n  [L-BFGS] fine-tuning for {cfg.epochs_lbfgs} steps …")
    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter,
        history_size=50, line_search_fn="strong_wolfe")

    for step in range(1, cfg.epochs_lbfgs + 1):
        def closure():
            lbfgs_opt.zero_grad()
            ua = model(X_dev)
            dl = F.mse_loss(ua, y_dev)
            ic = np.random.choice(N, cfg.n_colloc, replace=False)
            xs = torch.from_numpy(X_np[ic]).to(device).requires_grad_(True)
            pl = burgers_residual(model, xs, cfg.nu)
            ug = ua.reshape(Nt, Nx)
            px = (ug[:, 2:] - ug[:, :-2]) / (2.0 * dx_val)
            pt = (ug[2:, :] - ug[:-2, :]) / (2.0 * dt_val)
            sl = 0.5 * (
                F.l1_loss(px, du_x_dev[:, 1:-1]) /
                    (du_x_dev[:, 1:-1].abs().mean() + 1e-8) +
                F.l1_loss(pt, du_t_dev[1:-1, :]) /
                    (du_t_dev[1:-1, :].abs().mean() + 1e-8)
            )
            tot = dl + lambda_p * pl + cfg.sob_weight * sl
            tot.backward()
            return tot

        lbfgs_opt.step(closure)
        if step % max(1, cfg.epochs_lbfgs // 5) == 0 or step == 1:
            with torch.no_grad():
                dl_tmp = F.mse_loss(model(X_dev), y_dev).item()
            print(f"  L-BFGS step {step:5d} | Data MSE {dl_tmp:.3e}")

    # ── Eval ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        u_pred = model(X_dev).cpu().numpy()

    mse  = float(np.mean((u_pred - y_np) ** 2))
    rl2  = rel_l2(u_pred, y_np)
    w11  = w1_1_rel_error(u_pred, exact, x_vec, t_vec)

    return model, history, u_pred, mse, rl2, w11

# ── Plots ─────────────────────────────────────────────────────
def plot_results(t_vec, x_vec, exact, u_pred, rl2, history):
    nt, nx = exact.shape
    u2d    = u_pred.reshape(nt, nx)
    err    = np.abs(u2d - exact)

    # Solution panel
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)
    for col, (arr, title, cmap) in enumerate([
        (exact, "Exact $u(x,t)$",    "RdBu_r"),
        (u2d,   "Predicted $u(x,t)$","RdBu_r"),
        (err,   "Absolute Error",     "hot_r"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(x_vec, t_vec, arr, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, pad=0.02)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title, fontsize=11)
    for i, ti in enumerate([0, nt//4, nt//2]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x_vec, exact[ti], "k-", lw=2,   label="Exact")
        ax.plot(x_vec, u2d[ti],   "r--",lw=1.8, label="MLP+Sob")
        ax.set_title(f"t={t_vec[ti]:.3f}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"MLP+Sobolev  |  Rel L₂={rl2:.4f}", fontsize=13, y=1.01)
    plt.savefig("results/mlp_sob_solution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → results/mlp_sob_solution.png")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].semilogy(history["epoch"], history["data"],  color="#1f77b4", lw=1.8)
    axes[0].set_title("Data MSE"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log)"); axes[0].grid(alpha=0.25)
    axes[1].semilogy(history["epoch"], history["phys"],  color="#ff7f0e", lw=1.8)
    axes[1].set_title("Burgers PDE Residual"); axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[2].semilogy(history["epoch"], history["sob"],   color="#2ca02c", lw=1.8)
    axes[2].set_title("Sobolev Loss"); axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    fig.suptitle("MLP+Sobolev — Convergence", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/mlp_sob_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → results/mlp_sob_training.png")

# ── Main ──────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = get_device()

    X_np, y_np, t_vec, x_vec, exact = load_data()
    model, history, u_pred, mse, rl2, w11 = train(
        cfg, X_np, y_np, t_vec, x_vec, exact, device)

    print(f"\n  ✦ MSE     : {mse:.3e}")
    print(f"  ✦ Rel L2  : {rl2:.4f}")
    print(f"  ✦ W1,1    : {w11:.4f}")

    print("\n" + "═"*64)
    print("  COMPARISON TABLE")
    print("─"*64)
    rows = [
        ("MLP+Sobolev (this run)",   mse,    rl2,   w11),
        ("holo-Diff+Sob (full run)", 1.39e-6, 0.0019, 0.0018),
        ("MLP-Only (no Sobolev)",    6.70e-6, 0.0042, 0.0216),
        ("SA-PINN (literature)",     1.00e-5, 0.0032, None),
    ]
    print(f"  {'Run':<30} {'MSE':>12} {'Rel-L2':>10} {'W1,1':>10}")
    print("─"*64)
    for tag, m, r, w in rows:
        ws = f"{w:>10.4f}" if w is not None else "       N/A"
        print(f"  {tag:<30} {m:>12.3e} {r:>10.4f} {ws}")
    print("═"*64)
    print("\nInterpretation:")
    print("  If W1,1(MLP+Sob) ≈ W1,1(MLP-Only) >> W1,1(holo-Diff+Sob):")
    print("  → Sobolev alone cannot explain the improvement.")
    print("  → Holonomy is causally necessary for the W1,1 gain.")

    plot_results(t_vec, x_vec, exact, u_pred, rl2, history)
    print("\nDone. Results → results/")

if __name__ == "__main__":
    main()