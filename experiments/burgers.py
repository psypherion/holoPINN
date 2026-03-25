# ==============================================================
# Burgers holo-PINN 
# ==============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
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


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
class Config:
    nu               = 0.01 / np.pi
    seed             = 42

    # graph
    k_neighbors      = 6
    K_hops           = 2
    t_scale          = 2.0

    # network
    hidden           = 128
    n_layers         = 4
    n_fourier        = 64
    fourier_sigma    = 1.0          # default; overridden in σ-sweep

    # Adam phase
    epochs_adam      = 20_000
    lr               = 1e-3
    lr_min           = 1e-5

    # A1: L-BFGS phase
    epochs_lbfgs     = 5_000       # number of L-BFGS steps
    lbfgs_lr         = 1.0         # standard for L-BFGS
    lbfgs_max_iter   = 20          # per-step max line-search iterations

    # training
    n_colloc         = 512
    batch_size       = 2048
    log_every        = 1000
    grad_alpha       = 0.05

    # F1: LRA with hard cap
    lra_enabled      = True
    lra_update_every = 1000
    lra_alpha        = 0.9
    lra_lambda_max   = 100.0       
    phys_weight_init = 1.0

    # Sobolev
    sob_weight       = 0.05

    # A2: σ sweep epochs (shorter)
    sigma_sweep_epochs_adam  = 10_000
    sigma_sweep_epochs_lbfgs = 2_000
    sigma_sweep_values       = [0.5, 1.0, 2.0, 5.0]


# ──────────────────────────────────────────────────────────────
# Fourier MLP
# ──────────────────────────────────────────────────────────────
class FourierMLP(nn.Module):
    def __init__(self, n_fourier=64, sigma=1.0, hidden=128, n_layers=4):
        super().__init__()
        B = torch.randn(2, n_fourier) * sigma
        self.register_buffer("B", B)
        in_dim = 2 * n_fourier
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        proj = xt @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat)


# ──────────────────────────────────────────────────────────────
# holo GNN
# ──────────────────────────────────────────────────────────────
class holoInspiredGNN(nn.Module):
    def __init__(self, cfg, ablate=False, sigma_override=None):
        super().__init__()
        self.K      = cfg.K_hops
        self.ablate = ablate
        sigma = sigma_override if sigma_override is not None else cfg.fourier_sigma
        self.mlp    = FourierMLP(cfg.n_fourier, sigma,
                                 cfg.hidden, cfg.n_layers)
        self.log_w    = None
        self.edge_src = None
        self.edge_dst = None

    def build_graph(self, edge_index, device):
        self.edge_src = edge_index[0].to(device)
        self.edge_dst = edge_index[1].to(device)
        self.log_w    = nn.Parameter(
            torch.randn(edge_index.shape[1], device=device) * 0.1)

    def freeze_holonomy(self):
        self.log_w.requires_grad_(False)

    def mlp_forward(self, xt):
        return self.mlp(xt)

    def forward(self, xt):
        u = self.mlp(xt)
        if self.ablate:
            return u
        w   = torch.exp(self.log_w).unsqueeze(1)
        src, dst = self.edge_src, self.edge_dst
        for _ in range(self.K):
            msg   = w * u[src]
            u_new = torch.zeros_like(u)
            u_new.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            u = u + u_new
        return u


# ──────────────────────────────────────────────────────────────
# PDE residual on û (Fix 2 from v2, retained)
# ──────────────────────────────────────────────────────────────
def burgers_residual_full(model, X_dev, idx_c, nu):
    xt_c = X_dev[idx_c].detach().clone().requires_grad_(True)
    u_c  = model.mlp(xt_c)

    if not model.ablate:
        with torch.no_grad():
            u_all_det = model.mlp(X_dev)
        w   = torch.exp(model.log_w).unsqueeze(1)
        src, dst   = model.edge_src, model.edge_dst
        idx_c_t    = torch.from_numpy(idx_c).to(X_dev.device)
        node_map   = torch.full((X_dev.shape[0],), -1,
                                dtype=torch.long, device=X_dev.device)
        node_map[idx_c_t] = torch.arange(len(idx_c), device=X_dev.device)
        local_dst  = node_map[dst]
        valid      = local_dst >= 0
        e_src, e_dst, e_w = src[valid], local_dst[valid], w[valid]
        for _ in range(model.K):
            msg   = e_w * u_all_det[e_src]
            u_new = torch.zeros_like(u_c)
            u_new.scatter_add_(0, e_dst.unsqueeze(1).expand_as(msg), msg)
            u_c = u_c + u_new

    g    = torch.autograd.grad(u_c, xt_c,
                               grad_outputs=torch.ones_like(u_c),
                               create_graph=True)[0]
    u_x  = g[:, 0:1];  u_t = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_c,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    return torch.mean((u_t + u_c * u_x - nu * u_xx) ** 2)


def burgers_residual_mlp(model, xt_sub, nu):
    """Used for MLP-only ablation."""
    u    = model.mlp_forward(xt_sub)
    g    = torch.autograd.grad(u, xt_sub,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x  = g[:, 0:1];  u_t = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_sub,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    return torch.mean((u_t + u * u_x - nu * u_xx) ** 2)


# ──────────────────────────────────────────────────────────────
# F1+F2: LRA with cap and frozen-param guard
# ──────────────────────────────────────────────────────────────
def lra_update(model, loss_data, loss_pde, lambda_p, alpha, lambda_max):
    """
    F1: cap at lambda_max.
    F2: only use parameters that actually require grad.
        Holonomy freezes log_w → filter it out.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return lambda_p   # nothing to differentiate — skip

    try:
        grad_data = torch.autograd.grad(
            loss_data, trainable, retain_graph=True, allow_unused=True)
        grad_pde  = torch.autograd.grad(
            loss_pde,  trainable, retain_graph=True, allow_unused=True)
    except RuntimeError:
        return lambda_p   # safety fallback

    norm_data = sum(g.norm().item()**2
                    for g in grad_data if g is not None) ** 0.5
    norm_pde  = sum(g.norm().item()**2
                    for g in grad_pde  if g is not None) ** 0.5

    if norm_pde > 1e-10:
        hat       = norm_data / norm_pde
        lambda_p  = alpha * lambda_p + (1.0 - alpha) * hat
        lambda_p  = min(lambda_p, lambda_max)   # F1: hard cap

    return float(lambda_p)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def rel_l2(pred, exact):
    return float(np.linalg.norm(pred - exact) /
                 (np.linalg.norm(exact) + 1e-12))


def w1_1_rel_error(u_pred, exact, x_vec, t_vec):
    nt, nx = exact.shape
    u2d    = u_pred.reshape(nt, nx)
    dx     = float(x_vec[1] - x_vec[0])
    dt     = float(t_vec[1] - t_vec[0])
    diff   = u2d - exact
    num    = (np.mean(np.abs(diff)) +
              np.mean(np.abs(np.gradient(u2d,   dx, axis=1) -
                             np.gradient(exact, dx, axis=1))) +
              np.mean(np.abs(np.gradient(u2d,   dt, axis=0) -
                             np.gradient(exact, dt, axis=0))))
    den    = (np.mean(np.abs(exact)) +
              np.mean(np.abs(np.gradient(exact, dx, axis=1))) +
              np.mean(np.abs(np.gradient(exact, dt, axis=0))))
    return float(num / (den + 1e-12))


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
class DataLoader:
    def __init__(self, path="../datasets/Burgers.npz"):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at '{self.path}'.")
        data  = np.load(self.path)
        t_vec = data["t"].squeeze()
        x_vec = data["x"].squeeze()
        exact = data["usol"].T
        xx, tt = np.meshgrid(x_vec, t_vec)
        X = np.vstack([xx.ravel(), tt.ravel()]).T
        y = exact.reshape(-1, 1)
        print(f"[Data] t∈[{t_vec.min():.3f},{t_vec.max():.3f}]  "
              f"x∈[{x_vec.min():.3f},{x_vec.max():.3f}]  "
              f"Nt={len(t_vec)} Nx={len(x_vec)} N={X.shape[0]}")
        return X.astype(np.float32), y.astype(np.float32), t_vec, x_vec, exact


# ──────────────────────────────────────────────────────────────
# Training  (Adam phase + A1: L-BFGS phase)
# ──────────────────────────────────────────────────────────────
def train_model(cfg, X_np, y_np, edge_index, device,
                ablate=False, tag="", freeze_w=False,
                Nt=None, Nx=None,
                dx_val=None, dt_val=None,
                du_exact_grid=None, dt_exact_grid=None,
                epochs_adam_override=None,
                epochs_lbfgs_override=None,
                sigma_override=None):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # optionally override σ for sweep
    if sigma_override is not None:
        cfg_sigma = sigma_override
    else:
        cfg_sigma = cfg.fourier_sigma

    N     = X_np.shape[0]
    X_dev = torch.from_numpy(X_np).to(device)
    y_dev = torch.from_numpy(y_np).to(device)

    model = holoInspiredGNN(cfg, ablate=ablate, sigma_override=cfg_sigma).to(device)
    model.build_graph(edge_index, device)
    if freeze_w:
        model.freeze_holonomy()

    epochs_adam  = epochs_adam_override  or cfg.epochs_adam
    epochs_lbfgs = epochs_lbfgs_override or cfg.epochs_lbfgs
    use_sob      = (du_exact_grid is not None) and (dt_exact_grid is not None)
    lambda_p     = cfg.phys_weight_init

    mode_str = ("holonomy" if freeze_w else
                "ablation"       if ablate   else "diffusion")
    sob_str  = "+Sobolev" if use_sob else ""
    print(f"\n{'─'*64}")
    print(f"  Run: {tag}  |  mode={mode_str}{sob_str}  σ={cfg_sigma}")
    print(f"  hidden={cfg.hidden} layers={cfg.n_layers} "
          f"Adam={epochs_adam} L-BFGS={epochs_lbfgs}")
    print(f"{'─'*64}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_adam, eta_min=cfg.lr_min)

    history = {"data": [], "phys": [], "sob": [], "epoch": [], "lambda_p": []}

    # ── Adam phase ────────────────────────────────────────────
    for epoch in range(1, epochs_adam + 1):
        model.train()
        optimizer.zero_grad()

        u_all = model(X_dev)
        idx_b = torch.randint(0, N, (cfg.batch_size,), device=device)
        data_loss = F.mse_loss(u_all[idx_b], y_dev[idx_b])

        idx_c = np.random.choice(N, cfg.n_colloc, replace=False)
        if ablate:
            xt_s      = torch.from_numpy(X_np[idx_c]).to(device).requires_grad_(True)
            phys_loss = burgers_residual_mlp(model, xt_s, cfg.nu)
        else:
            phys_loss = burgers_residual_full(model, X_dev, idx_c, cfg.nu)

        # F1+F2: LRA with cap and frozen-param guard
        if (cfg.lra_enabled and
                epoch % cfg.lra_update_every == 0 and epoch > 1):
            lambda_p = lra_update(
                model, data_loss, phys_loss,
                lambda_p, cfg.lra_alpha, cfg.lra_lambda_max)

        sob_loss = torch.zeros(1, device=device)
        if use_sob:
            u_grid    = u_all.reshape(Nt, Nx)
            dpx       = (u_grid[:, 2:] - u_grid[:, :-2]) / (2.0 * dx_val)
            dpt       = (u_grid[2:, :] - u_grid[:-2, :]) / (2.0 * dt_val)
            sob_loss  = 0.5 * (
                F.l1_loss(dpx, du_exact_grid[:, 1:-1]) /
                    (du_exact_grid[:, 1:-1].abs().mean() + 1e-8) +
                F.l1_loss(dpt, dt_exact_grid[1:-1, :]) /
                    (dt_exact_grid[1:-1, :].abs().mean() + 1e-8)
            )

        loss = data_loss + lambda_p * phys_loss + cfg.sob_weight * sob_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["data"].append(data_loss.item())
        history["phys"].append(phys_loss.item())
        history["sob"].append(sob_loss.item() if use_sob else 0.0)
        history["epoch"].append(epoch)
        history["lambda_p"].append(lambda_p)

        if epoch % cfg.log_every == 0 or epoch == 1:
            s = (f"| Sob {sob_loss.item():.2e} " if use_sob else "")
            print(f"  Epoch {epoch:6d} | Data {data_loss.item():.3e} "
                  f"| PDE {phys_loss.item():.3e} {s}"
                  f"| λ_p {lambda_p:.2f} | LR {scheduler.get_last_lr()[0]:.2e}")

    # ── A1: L-BFGS fine-tuning phase ─────────────────────────
    if epochs_lbfgs > 0:
        print(f"\n  [L-BFGS] fine-tuning for {epochs_lbfgs} steps …")

        # L-BFGS needs a closure
        lbfgs_opt = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        lbfgs_losses = []

        for step in range(1, epochs_lbfgs + 1):
            def closure():
                lbfgs_opt.zero_grad()
                u_a = model(X_dev)
                dl  = F.mse_loss(u_a, y_dev)

                ic  = np.random.choice(N, cfg.n_colloc, replace=False)
                if ablate:
                    xs  = torch.from_numpy(X_np[ic]).to(device).requires_grad_(True)
                    pl  = burgers_residual_mlp(model, xs, cfg.nu)
                else:
                    pl  = burgers_residual_full(model, X_dev, ic, cfg.nu)

                sl  = torch.zeros(1, device=device)
                if use_sob:
                    ug  = u_a.reshape(Nt, Nx)
                    px  = (ug[:, 2:] - ug[:, :-2]) / (2.0 * dx_val)
                    pt  = (ug[2:, :] - ug[:-2, :]) / (2.0 * dt_val)
                    sl  = 0.5 * (
                        F.l1_loss(px, du_exact_grid[:, 1:-1]) /
                            (du_exact_grid[:, 1:-1].abs().mean() + 1e-8) +
                        F.l1_loss(pt, dt_exact_grid[1:-1, :]) /
                            (dt_exact_grid[1:-1, :].abs().mean() + 1e-8)
                    )
                tot = dl + lambda_p * pl + cfg.sob_weight * sl
                tot.backward()
                return tot

            lbfgs_opt.step(closure)

            if step % max(1, epochs_lbfgs // 5) == 0 or step == 1:
                with torch.no_grad():
                    u_tmp  = model(X_dev)
                    dl_tmp = F.mse_loss(u_tmp, y_dev).item()
                lbfgs_losses.append(dl_tmp)
                print(f"  L-BFGS step {step:5d} | Data MSE {dl_tmp:.3e}")

        history["lbfgs_data"] = lbfgs_losses

    # ── Evaluation ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        u_pred = model(X_dev).cpu().numpy()

    mse  = float(np.mean((u_pred - y_np) ** 2))
    rl2  = rel_l2(u_pred, y_np)
    w_np = (torch.exp(model.log_w).detach().cpu().numpy()
            if (not ablate and model.log_w is not None) else None)

    print(f"\n  ✦ MSE     : {mse:.3e}")
    print(f"  ✦ Rel L2  : {rl2:.4f}")
    return model, history, u_pred, w_np, mse, rl2


# ──────────────────────────────────────────────────────────────
# Plots  (same API as v2)
# ──────────────────────────────────────────────────────────────
def plot_solution(t_vec, x_vec, exact, u_pred, rl2, tag, fname):
    nt, nx = exact.shape
    u2d    = u_pred.reshape(nt, nx)
    err    = np.abs(u2d - exact)
    fig    = plt.figure(figsize=(18, 10))
    gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)
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
        ax.plot(x_vec, u2d[ti],   "r--",lw=1.8, label="Pred")
        ax.set_title(f"t={t_vec[ti]:.3f}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f"{tag}  |  Rel L₂={rl2:.4f}", fontsize=13, y=1.01)
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_shock_slices(t_vec, x_vec, exact, pred_list, label_list, rl2_list, fname):
    nt, nx  = exact.shape
    colors  = ["r","g","b","purple","orange"]
    styles  = ["--","-.",":",(0,(3,1,1,1)),(0,(5,1))]
    fig, axes = plt.subplots(1, 4, figsize=(18,4), sharey=True)
    for ax, ti in zip(axes, [nt//2, 3*nt//4, 7*nt//8, nt-1]):
        ax.plot(x_vec, exact[ti], "k-", lw=2.5, label="Exact", zorder=10)
        for i,(p,l,r) in enumerate(zip(pred_list,label_list,rl2_list)):
            ax.plot(x_vec, p.reshape(nt,nx)[ti],
                    color=colors[i%len(colors)], ls=styles[i%len(styles)],
                    lw=1.8, label=f"{l} ({r:.3f})", zorder=9-i)
        ax.set_title(f"t={t_vec[ti]:.3f}"); ax.set_xlabel("x")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    axes[0].set_ylabel("u(x,t)")
    fig.suptitle("Late-Time Shock — All Runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_training_curves(histories, labels, fname):
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    styles = ["-","-","-.","--",":"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for i,(h,lab) in enumerate(zip(histories,labels)):
        axes[0].semilogy(h["epoch"], h["data"],
                         color=colors[i], lw=1.8, ls=styles[i], label=lab)
        axes[1].semilogy(h["epoch"], h["phys"],
                         color=colors[i], lw=1.8, ls=styles[i], label=lab)
        axes[2].plot(h["epoch"], h["lambda_p"],
                     color=colors[i], lw=1.8, ls=styles[i], label=lab)
    for ax,title in zip(axes[:2],["Data MSE","Burgers PDE Residual"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("λ_p (capped at 100)")
    axes[2].set_title("LRA Physics Weight  — capped)"); axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Convergence — All Runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_holonomy_spectrum(w_vals, tag, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(w_vals, bins=250, color="#1f77b4",
                 edgecolor="none", log=True, alpha=0.85)
    axes[0].axvline(np.median(w_vals), color="k", ls="--",
                    lw=1.4, label=f"median={np.median(w_vals):.3f}")
    axes[0].set_xlabel(r"$e^{\theta_{ij}}$"); axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Holonomy Weight Dist."); axes[0].legend()
    pcts=[1,5,25,50,75,95,99]; vals=np.percentile(w_vals,pcts)
    bars=axes[1].barh([f"{p}th" for p in pcts],vals,color="#1f77b4",alpha=0.8)
    for bar,val in zip(bars,vals):
        axes[1].text(val*1.01,bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}",va="center",fontsize=8)
    axes[1].set_xlabel(r"$e^{\theta_{ij}}$"); axes[1].set_title("Percentiles")
    fig.suptitle(f"Holonomy Spectrum — {tag}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_holonomy_spatial(model, X_np, edge_index, tag, fname):
    w_vals = torch.exp(model.log_w).detach().cpu().numpy()
    src    = edge_index[0].cpu().numpy()
    dst    = edge_index[1].cpu().numpy()
    mid_x  = 0.5*(X_np[src,0]+X_np[dst,0])
    mid_t  = 0.5*(X_np[src,1]+X_np[dst,1])
    vmax   = np.percentile(w_vals, 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(mid_x, mid_t, c=w_vals,
                         cmap="hot", s=0.3, alpha=0.35,
                         vmin=0, vmax=vmax, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label=r"$e^{\theta_{ij}}$")
    axes[0].axvline(0.0, color="cyan", lw=1.8, ls="--", label="shock x=0")
    axes[0].axhline(0.40, color="lime", lw=1.2, ls=":", label="shock onset")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Holonomy weights in (x,t)"); axes[0].legend(fontsize=8)

    shock_mask = (np.abs(mid_x)<0.15) & (mid_t>0.40)
    w_s  = w_vals[shock_mask];  w_sm = w_vals[~shock_mask]
    means= [w_s.mean(), w_sm.mean()]; stds=[w_s.std(), w_sm.std()]
    bars = axes[1].bar(["Shock corridor","Smooth region"],
                       means, yerr=stds,
                       color=["#d62728","#1f77b4"],
                       alpha=0.85, edgecolor="k", capsize=6)
    for bar,val in zip(bars,means):
        axes[1].text(bar.get_x()+bar.get_width()/2, val*1.04,
                     f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ratio = means[0]/(means[1]+1e-12)
    axes[1].text(0.98, 0.95, f"Shock/Smooth = {ratio:.2f}×",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    axes[1].set_ylabel(r"Mean $e^{\theta_{ij}}$")
    axes[1].set_title("Shock vs. Smooth"); axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle(f"Spatial Holonomy — {tag}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")
    print(f"  [Spatial] shock={means[0]:.3f}  smooth={means[1]:.3f}  ratio={ratio:.2f}×")


def plot_sigma_sweep(sigma_results, fname):
    """A2: bar chart of Rel-L² vs σ for holo-Diff+Sob."""
    sigmas = [r[0] for r in sigma_results]
    rl2s   = [r[1] for r in sigma_results]
    mses   = [r[2] for r in sigma_results]
    w11s   = [r[3] for r in sigma_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [f"σ={s}" for s in sigmas]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    for ax, vals, ylabel, title in zip(
        axes,
        [rl2s, mses, w11s],
        ["Rel-L²","MSE","W1,1"],
        ["Rel-L² vs σ","MSE vs σ","W1,1 vs σ"]
    ):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor="k", lw=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val*1.03,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, max(vals)*1.3)
        ax.grid(alpha=0.25, axis="y")
    best_idx = int(np.argmin(rl2s))
    fig.suptitle(
        f"Fourier σ Sweep — holo-Diff+Sob  |  Best: σ={sigmas[best_idx]}"
        f" (Rel-L²={rl2s[best_idx]:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")
    return sigmas[best_idx]


def plot_metrics_bar(results, fname):
    tags   = [r[0] for r in results]
    rl2s   = [r[2] for r in results]
    mses   = [r[1] for r in results]
    w11s   = [r[3] for r in results]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"][:len(results)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, vals, ylabel, title in zip(
        axes,
        [rl2s, mses, w11s],
        ["Relative L2","MSE","W1,1 Rel Error"],
        ["Rel L2","MSE","W1,1"]
    ):
        bars = ax.bar(tags, vals, color=colors, alpha=0.85, edgecolor="k", lw=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val*1.03,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, max(vals)*1.28); ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Full Ablation Study", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def summary_table(results):
    print("\n" + "═"*72)
    print(f"  {'Run':<30} {'MSE':>12} {'Rel L2':>10} {'W1,1 Rel':>12}")
    print("─"*72)
    for tag, mse, rl2, w11 in results:
        print(f"  {tag:<30} {mse:>12.3e} {rl2:>10.4f} {w11:>12.4f}")
    print("─"*72)
    print("  v8 reference:")
    for tag, mse, rl2, w11 in [
        ("holo-Diff+Sob (v8)",  9.45e-4, 0.0500, 0.0961),
        ("MLP-Only (v8)",      7.89e-3, 0.1446, 0.3150),
        ("SA-PINN (literature)",1.0e-5, 0.0032, None),
    ]:
        w_str = f"{w11:>12.4f}" if w11 else "          N/A"
        print(f"  {tag:<30} {mse:>12.3e} {rl2:>10.4f} {w_str}")
    print("═"*72)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = get_device()

    X_np, y_np, t_vec, x_vec, exact = DataLoader().load()
    N  = X_np.shape[0]
    Nt = len(t_vec)
    Nx = len(x_vec)

    # Fix 3 (from v2): scaled kNN
    X_scaled = X_np.copy(); X_scaled[:, 1] *= cfg.t_scale
    print(f"\n[Graph] Building k={cfg.k_neighbors} kNN on scaled (t×{cfg.t_scale})…")
    tree = KDTree(X_scaled)
    _, nb = tree.query(X_scaled, k=cfg.k_neighbors + 1)
    rows = np.repeat(np.arange(N), cfg.k_neighbors)
    cols = nb[:, 1:].ravel()
    edge_index = torch.LongTensor(np.vstack([rows, cols]))
    print(f"[Graph] edges = {edge_index.shape[1]}")

    dx_val = float(x_vec[1] - x_vec[0])
    dt_val = float(t_vec[1] - t_vec[0])
    du_exact_2d  = np.gradient(exact, dx_val, axis=1).astype(np.float32)
    dt_exact_2d  = np.gradient(exact, dt_val, axis=0).astype(np.float32)
    du_exact_grid = torch.from_numpy(du_exact_2d).to(device)
    dt_exact_grid = torch.from_numpy(dt_exact_2d).to(device)

    results, histories = [], []

    # ══════════════════════════════════════════
    # A2: σ sweep  (holo-Diff+Sob only, shorter)
    # ══════════════════════════════════════════
    print("\n" + "═"*64)
    print("  A2: Fourier σ Sweep — holo-Diff+Sob")
    print("═"*64)
    sigma_results = []
    for sigma in cfg.sigma_sweep_values:
        tag_s = f"σ-sweep σ={sigma}"
        ms, hs, ps, ws, mse_s, rl2_s = train_model(
            cfg, X_np, y_np, edge_index, device,
            ablate=False, tag=tag_s,
            Nt=Nt, Nx=Nx, dx_val=dx_val, dt_val=dt_val,
            du_exact_grid=du_exact_grid, dt_exact_grid=dt_exact_grid,
            epochs_adam_override=cfg.sigma_sweep_epochs_adam,
            epochs_lbfgs_override=cfg.sigma_sweep_epochs_lbfgs,
            sigma_override=sigma,
        )
        w11_s = w1_1_rel_error(ps, exact, x_vec, t_vec)
        print(f"  σ={sigma}: MSE={mse_s:.3e}  Rel-L²={rl2_s:.4f}  W1,1={w11_s:.4f}")
        sigma_results.append((sigma, rl2_s, mse_s, w11_s))

    best_sigma = plot_sigma_sweep(sigma_results, "results/sigma_sweep.png")
    print(f"\n  Best σ = {best_sigma}")
    cfg.fourier_sigma = best_sigma   # use best σ for main ablation runs

    # ══════════════════════════════════════════
    # RUN 1 — holo-Diffusion
    # ══════════════════════════════════════════
    m1, h1, pred1, w1, mse1, rl2_1 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="holo-Diffusion"
    )
    w11_1 = w1_1_rel_error(pred1, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_1:.4f}")
    results.append(("holo-Diffusion", mse1, rl2_1, w11_1))
    histories.append(h1)
    plot_solution(t_vec, x_vec, exact, pred1, rl2_1,
                  "holo-Diffusion", "results/holo_diff_solution.png")
    plot_holonomy_spectrum(w1, "Diffusion", "results/holonomy_spectrum_diff.png")
    plot_holonomy_spatial(m1, X_np, edge_index,
                          "holo-Diffusion", "results/holonomy_spatial_diff.png")

    # ══════════════════════════════════════════
    # RUN 2 — holo-Diff+Sob  (best variant)
    # ══════════════════════════════════════════
    m2, h2, pred2, w2, mse2, rl2_2 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="holo-Diff+Sob",
        Nt=Nt, Nx=Nx, dx_val=dx_val, dt_val=dt_val,
        du_exact_grid=du_exact_grid, dt_exact_grid=dt_exact_grid
    )
    w11_2 = w1_1_rel_error(pred2, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_2:.4f}")
    results.append(("holo-Diff+Sob", mse2, rl2_2, w11_2))
    histories.append(h2)
    plot_solution(t_vec, x_vec, exact, pred2, rl2_2,
                  "holo-Diff+Sob", "results/holo_sob_solution.png")
    plot_holonomy_spectrum(w2, "Diff+Sob", "results/holonomy_spectrum_sob.png")
    plot_holonomy_spatial(m2, X_np, edge_index,
                          "holo-Diff+Sob", "results/holonomy_spatial_sob.png")

    # ══════════════════════════════════════════
    # RUN 3 — MLP-Only  (Fourier baseline)
    # ══════════════════════════════════════════
    m3, h3, pred3, _, mse3, rl2_3 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=True, tag="MLP-Only"
    )
    w11_3 = w1_1_rel_error(pred3, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_3:.4f}")
    results.append(("MLP-Only (Fourier)", mse3, rl2_3, w11_3))
    histories.append(h3)
    plot_solution(t_vec, x_vec, exact, pred3, rl2_3,
                  "MLP-Only (Fourier)", "results/mlp_solution.png")

    # ══════════════════════════════════════════
    # RUN 4 — Holonomy null ablation
    #   F2: LRA skips frozen params — no crash
    # ══════════════════════════════════════════
    m4, h4, pred4, w4, mse4, rl2_4 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="Holonomy", freeze_w=True
    )
    w11_4 = w1_1_rel_error(pred4, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_4:.4f}")
    results.append(("Holonomy", mse4, rl2_4, w11_4))
    histories.append(h4)
    plot_solution(t_vec, x_vec, exact, pred4, rl2_4,
                  "Holonomy", "results/fixed_holo_solution.png")
    plot_holonomy_spectrum(w4, "Fixed (frozen)", "results/holonomy_spectrum_fixed.png")
    plot_holonomy_spatial(m4, X_np, edge_index,
                          "Holonomy", "results/holonomy_spatial_fixed.png")

    # ══════════════════════════════════════════
    # Combined plots
    # ══════════════════════════════════════════
    all_preds  = [pred1, pred2, pred3, pred4]
    all_labels = ["holo-Diff","holo-Diff+Sob","MLP-Only","Fixed"]
    all_rl2s   = [rl2_1, rl2_2, rl2_3, rl2_4]

    plot_training_curves(histories, all_labels, "results/training_curves.png")
    plot_shock_slices(t_vec, x_vec, exact, all_preds, all_labels, all_rl2s,
                      "results/shock_slices.png")
    plot_metrics_bar(results, "results/ablation_bar.png")

    summary_table(results)
    print("\nAll results → results/")


if __name__ == "__main__":
    main()