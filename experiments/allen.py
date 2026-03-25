# ==============================================================
# Allen-Cahn PINN
#
# PDE:  u_t - d*u_xx - 5*(u - u³) = 0
#       d = 0.001,  x ∈ [-1,1],  t ∈ [0,1]
# Interface develops at x=0; holonomy weights should
# concentrate there — the "R_interface" analogue of R_shock.
#
# Architecture identical to holoPINN_v3 (Burgers).
# Runs:
#   R1: Diffusion
#   R2: Diff+Sob  (best variant)
#   R3: MLP-Only      (Fourier baseline)
#   R4: Fixed-Holonomy (null ablation)
# ==============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("results_ac", exist_ok=True)


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
# Config  — same as v3, only PDE constants change
# ──────────────────────────────────────────────────────────────
class Config:
    # Allen-Cahn constants
    d               = 0.001        # diffusivity
    ac_coeff        = 5.0          # coefficient of (u - u³)

    seed            = 42
    k_neighbors     = 6
    K_hops          = 2
    t_scale         = 2.0          # scale t before kNN

    hidden          = 128
    n_layers        = 4
    n_fourier       = 64
    fourier_sigma   = 1.0          # best σ from Burgers sweep

    epochs_adam     = 20_000
    lr              = 1e-3
    lr_min          = 1e-5
    epochs_lbfgs    = 5_000
    lbfgs_lr        = 1.0
    lbfgs_max_iter  = 20

    n_colloc        = 512
    batch_size      = 2048
    log_every       = 1000
    grad_alpha      = 0.05

    lra_enabled      = True
    lra_update_every = 1000
    lra_alpha        = 0.9
    lra_lambda_max   = 100.0
    phys_weight_init = 1.0

    sob_weight      = 0.05

    # Interface corridor for R_interface metric
    # Allen-Cahn develops a sharp interface at x≈0 for t>0.3
    interface_x_half = 0.15
    interface_t_min  = 0.30


# ──────────────────────────────────────────────────────────────
# Data loading  (exact gen_testdata from user prompt)
# ──────────────────────────────────────────────────────────────
def gen_testdata(path="../datasets/Allen_Cahn.mat"):
    if not os.path.exists(path):
        # try local path fallback
        alt = "dataset/Allen_Cahn.mat"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Allen_Cahn.mat not found at '{path}' or '{alt}'.\n"
                "Place the file in dataset/ and rerun."
            )
    data = loadmat(path)
    t    = data["t"]
    x    = data["x"]
    u    = data["u"]

    dt = dx = 0.01          # as specified
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]

    # recover grid dimensions for reshaping
    Nt = t.flatten().shape[0]
    Nx = x.flatten().shape[0]
    u_grid = u   # shape depends on mat layout — we reshape later

    print(f"[Data] t∈[{t.min():.3f},{t.max():.3f}]  "
          f"x∈[{x.min():.3f},{x.max():.3f}]  "
          f"Nt={Nt}  Nx={Nx}  N={X.shape[0]}")
    return (X.astype(np.float32),
            y.astype(np.float32),
            t.flatten(), x.flatten(),
            u_grid, Nt, Nx, dx, dt)


# ──────────────────────────────────────────────────────────────
# Fourier MLP  (identical to v3)
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
# LQG GNN  (identical to v3)
# ──────────────────────────────────────────────────────────────
class LQGInspiredGNN(nn.Module):
    def __init__(self, cfg, ablate=False, sigma_override=None):
        super().__init__()
        self.K      = cfg.K_hops
        self.ablate = ablate
        sigma = sigma_override if sigma_override is not None else cfg.fourier_sigma
        self.mlp    = FourierMLP(cfg.n_fourier, sigma, cfg.hidden, cfg.n_layers)
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
# Allen-Cahn PDE residual  ← KEY CHANGE FROM BURGERS
# r = u_t - d*u_xx - 5*(u - u³)
# ──────────────────────────────────────────────────────────────
def allen_cahn_residual_full(model, X_dev, idx_c, cfg):
    """PDE residual on û = H_K(φ_ψ) at collocation nodes idx_c."""
    xt_c = X_dev[idx_c].detach().clone().requires_grad_(True)
    u_c  = model.mlp(xt_c)

    if not model.ablate:
        with torch.no_grad():
            u_all_det = model.mlp(X_dev)
        w          = torch.exp(model.log_w).unsqueeze(1)
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

    # Derivatives of û w.r.t. (x, t)
    g    = torch.autograd.grad(u_c, xt_c,
                               grad_outputs=torch.ones_like(u_c),
                               create_graph=True)[0]
    u_x  = g[:, 0:1]
    u_t  = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_c,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]

    # Allen-Cahn: u_t - d*u_xx - 5*(u - u³) = 0
    r = u_t - cfg.d * u_xx - cfg.ac_coeff * (u_c - u_c**3)
    return torch.mean(r**2)


def allen_cahn_residual_mlp(model, xt_sub, cfg):
    """PDE residual on bare MLP (MLP-only ablation run)."""
    u    = model.mlp_forward(xt_sub)
    g    = torch.autograd.grad(u, xt_sub,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x  = g[:, 0:1]
    u_t  = g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_sub,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    r = u_t - cfg.d * u_xx - cfg.ac_coeff * (u - u**3)
    return torch.mean(r**2)


# ──────────────────────────────────────────────────────────────
# LRA  (identical to v3, frozen-param safe)
# ──────────────────────────────────────────────────────────────
def lra_update(model, loss_data, loss_pde, lambda_p, alpha, lambda_max):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return lambda_p
    try:
        grad_data = torch.autograd.grad(
            loss_data, trainable, retain_graph=True, allow_unused=True)
        grad_pde  = torch.autograd.grad(
            loss_pde,  trainable, retain_graph=True, allow_unused=True)
    except RuntimeError:
        return lambda_p
    norm_data = sum(g.norm().item()**2 for g in grad_data if g is not None)**0.5
    norm_pde  = sum(g.norm().item()**2 for g in grad_pde  if g is not None)**0.5
    if norm_pde > 1e-10:
        hat      = norm_data / norm_pde
        lambda_p = alpha * lambda_p + (1.0 - alpha) * hat
        lambda_p = min(lambda_p, lambda_max)
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
# Training  (Adam + L-BFGS, identical structure to v3)
# ──────────────────────────────────────────────────────────────
def train_model(cfg, X_np, y_np, edge_index, device,
                ablate=False, tag="", freeze_w=False,
                Nt=None, Nx=None,
                dx_val=None, dt_val=None,
                du_exact_grid=None, dt_exact_grid=None):

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    N     = X_np.shape[0]
    X_dev = torch.from_numpy(X_np).to(device)
    y_dev = torch.from_numpy(y_np).to(device)

    model = LQGInspiredGNN(cfg, ablate=ablate).to(device)
    model.build_graph(edge_index, device)
    if freeze_w:
        model.freeze_holonomy()

    use_sob  = (du_exact_grid is not None) and (dt_exact_grid is not None)
    lambda_p = cfg.phys_weight_init

    mode_str = ("fixed-holonomy" if freeze_w else
                "ablation"       if ablate   else "diffusion")
    sob_tag  = "+Sobolev" if use_sob else ""
    print(f"\n{'─'*64}")
    print(f"  Run: {tag}  |  mode={mode_str}{sob_tag}")
    print(f"  d={cfg.d}  ac_coeff={cfg.ac_coeff}  "
          f"σ={cfg.fourier_sigma}  epochs={cfg.epochs_adam}+{cfg.epochs_lbfgs}")
    print(f"{'─'*64}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs_adam, eta_min=cfg.lr_min)

    history = {"data": [], "phys": [], "sob": [], "epoch": [], "lambda_p": []}

    # ── Adam ──────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs_adam + 1):
        model.train()
        optimizer.zero_grad()

        u_all = model(X_dev)
        idx_b = torch.randint(0, N, (cfg.batch_size,), device=device)
        data_loss = F.mse_loss(u_all[idx_b], y_dev[idx_b])

        idx_c = np.random.choice(N, cfg.n_colloc, replace=False)
        if ablate:
            xt_s      = torch.from_numpy(X_np[idx_c]).to(device).requires_grad_(True)
            phys_loss = allen_cahn_residual_mlp(model, xt_s, cfg)
        else:
            phys_loss = allen_cahn_residual_full(model, X_dev, idx_c, cfg)

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
            s = f"| Sob {sob_loss.item():.2e} " if use_sob else ""
            print(f"  Epoch {epoch:6d} | Data {data_loss.item():.3e} "
                  f"| PDE {phys_loss.item():.3e} {s}"
                  f"| λ_p {lambda_p:.2f} | LR {scheduler.get_last_lr()[0]:.2e}")

    # ── L-BFGS ────────────────────────────────────────────────
    if cfg.epochs_lbfgs > 0:
        print(f"\n  [L-BFGS] fine-tuning for {cfg.epochs_lbfgs} steps …")
        lbfgs_opt = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter,
            history_size=50, line_search_fn="strong_wolfe")
        lbfgs_losses = []

        for step in range(1, cfg.epochs_lbfgs + 1):
            def closure():
                lbfgs_opt.zero_grad()
                u_a  = model(X_dev)
                dl   = F.mse_loss(u_a, y_dev)
                ic   = np.random.choice(N, cfg.n_colloc, replace=False)
                if ablate:
                    xs  = torch.from_numpy(X_np[ic]).to(device).requires_grad_(True)
                    pl  = allen_cahn_residual_mlp(model, xs, cfg)
                else:
                    pl  = allen_cahn_residual_full(model, X_dev, ic, cfg)
                sl   = torch.zeros(1, device=device)
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
            if step % max(1, cfg.epochs_lbfgs // 5) == 0 or step == 1:
                with torch.no_grad():
                    dl_tmp = F.mse_loss(model(X_dev), y_dev).item()
                lbfgs_losses.append(dl_tmp)
                print(f"  L-BFGS step {step:5d} | Data MSE {dl_tmp:.3e}")

    model.eval()
    with torch.no_grad():
        u_pred = model(X_dev).cpu().numpy()

    mse  = float(np.mean((u_pred - y_np)**2))
    rl2  = rel_l2(u_pred, y_np)
    w_np = (torch.exp(model.log_w).detach().cpu().numpy()
            if not ablate else None)

    print(f"\n  ✦ MSE     : {mse:.3e}")
    print(f"  ✦ Rel L2  : {rl2:.4f}")
    return model, history, u_pred, w_np, mse, rl2


# ──────────────────────────────────────────────────────────────
# Plots
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
        ax.plot(x_vec, u2d[ti],   "r--",lw=1.8, label="PINN")
        ax.set_title(f"t={t_vec[ti]:.3f}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlabel("x"); ax.set_ylabel("u")
    fig.suptitle(f"{tag}  |  Rel L₂={rl2:.4f}", fontsize=13, y=1.01)
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_interface_slices(t_vec, x_vec, exact, pred_list, label_list, rl2_list, fname):
    """Cross-sections at late times around the sharp interface."""
    nt, nx  = exact.shape
    colors  = ["r","b","purple","orange"]
    styles  = ["--","-.",":",(0,(3,1,1,1))]
    t_idx   = [nt//2, 3*nt//4, 7*nt//8, nt-1]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, ti in zip(axes, t_idx):
        ax.plot(x_vec, exact[ti], "k-", lw=2.5, label="Exact", zorder=10)
        for i,(p,l,r) in enumerate(zip(pred_list,label_list,rl2_list)):
            ax.plot(x_vec, p.reshape(nt,nx)[ti],
                    color=colors[i%len(colors)], ls=styles[i%len(styles)],
                    lw=1.8, label=f"{l} ({r:.3f})", zorder=9-i)
        ax.set_title(f"t={t_vec[ti]:.3f}"); ax.set_xlabel("x")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    axes[0].set_ylabel("u(x,t)")
    fig.suptitle("Allen-Cahn: Interface Cross-Sections — All Runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_training_curves(histories, labels, fname):
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    styles = ["-","-","-.","--"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for i,(h,lab) in enumerate(zip(histories,labels)):
        axes[0].semilogy(h["epoch"], h["data"],
                         color=colors[i], lw=1.8, ls=styles[i], label=lab)
        axes[1].semilogy(h["epoch"], h["phys"],
                         color=colors[i], lw=1.8, ls=styles[i], label=lab)
        axes[2].plot(h["epoch"], h["lambda_p"],
                     color=colors[i], lw=1.8, ls=styles[i], label=lab)
    for ax, title in zip(axes[:2], ["Data MSE","Allen-Cahn PDE Residual"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("λ_p (capped at 100)")
    axes[2].set_title("LRA Physics Weight"); axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle("Allen-Cahn: Convergence — All Runs", fontsize=13)
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
    axes[0].set_title("Holonomy Weight Distribution"); axes[0].legend()
    pcts=[1,5,25,50,75,95,99]; vals=np.percentile(w_vals,pcts)
    bars=axes[1].barh([f"{p}th" for p in pcts],vals,color="#1f77b4",alpha=0.8)
    for bar,val in zip(bars,vals):
        axes[1].text(val*1.01,bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}",va="center",fontsize=8)
    axes[1].set_xlabel(r"$e^{\theta_{ij}}$"); axes[1].set_title("Percentiles")
    fig.suptitle(f"Allen-Cahn Holonomy Spectrum — {tag}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fname}")


def plot_holonomy_spatial(model, X_np, edge_index, cfg, tag, fname):
    """
    Spatial holonomy analysis for Allen-Cahn.
    Interface corridor: |x| < interface_x_half, t > interface_t_min
    Analogous to shock corridor in Burgers.
    R_interface = mean(w in corridor) / mean(w outside)
    """
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
    axes[0].axvline(0.0,  color="cyan", lw=1.8, ls="--", label="interface x=0")
    axes[0].axhline(cfg.interface_t_min, color="lime",
                    lw=1.2, ls=":", label=f"onset t={cfg.interface_t_min}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Holonomy weights in (x,t)"); axes[0].legend(fontsize=8)

    iface_mask = (np.abs(mid_x) < cfg.interface_x_half) & (mid_t > cfg.interface_t_min)
    w_i   = w_vals[iface_mask]
    w_rest= w_vals[~iface_mask]
    means = [w_i.mean() if len(w_i)>0 else 0.0,
             w_rest.mean() if len(w_rest)>0 else 0.0]
    stds  = [w_i.std()  if len(w_i)>0 else 0.0,
             w_rest.std() if len(w_rest)>0 else 0.0]
    bars  = axes[1].bar(["Interface corridor","Smooth region"],
                        means, yerr=stds,
                        color=["#d62728","#1f77b4"],
                        alpha=0.85, edgecolor="k", capsize=6)
    for bar,val in zip(bars,means):
        axes[1].text(bar.get_x()+bar.get_width()/2, val*1.04,
                     f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ratio = means[0]/(means[1]+1e-12)
    axes[1].text(0.98, 0.95, f"Interface/Smooth = {ratio:.2f}×",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    axes[1].set_ylabel(r"Mean $e^{\theta_{ij}}$")
    axes[1].set_title("Interface vs. Smooth Region"); axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle(f"Allen-Cahn Spatial Holonomy — {tag}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    r_str = f"{ratio:.2f}×"
    print(f"  Saved → {fname}")
    print(f"  [Spatial] interface={means[0]:.3f}  smooth={means[1]:.3f}  "
          f"R_interface={r_str}")
    return ratio


def plot_metrics_bar(results, fname):
    tags   = [r[0] for r in results]
    rl2s   = [r[2] for r in results]
    mses   = [r[1] for r in results]
    w11s   = [r[3] for r in results]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"][:len(results)]
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
    fig.suptitle("Allen-Cahn: Full Ablation Study", fontsize=13)
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
    print("  Burgers v3 reference (Diff+Sob):")
    print(f"  {'Diff+Sob Burgers':<30} {'~5.7e-07':>12} {'0.0012':>10} {'0.0031':>12}")
    print("═"*72)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = get_device()

    (X_np, y_np,
     t_vec, x_vec,
     u_grid_raw, Nt, Nx,
     dx_val, dt_val) = gen_testdata()

    # Reshape exact solution to (Nt, Nx)
    # Allen_Cahn.mat: u is stored as (Nx, Nt) in most versions
    # We check and transpose if needed
    if u_grid_raw.shape == (Nx, Nt):
        exact = u_grid_raw.T          # → (Nt, Nx)
    elif u_grid_raw.shape == (Nt, Nx):
        exact = u_grid_raw
    else:
        # fallback: infer from flattened y
        exact = y_np.reshape(Nt, Nx)
    print(f"[Grid] exact.shape = {exact.shape}  (Nt={Nt}, Nx={Nx})")

    # Scaled kNN graph
    X_scaled = X_np.copy(); X_scaled[:, 1] *= cfg.t_scale
    N = X_np.shape[0]
    print(f"\n[Graph] Building k={cfg.k_neighbors} kNN on scaled (t×{cfg.t_scale})…")
    tree   = KDTree(X_scaled)
    _, nb  = tree.query(X_scaled, k=cfg.k_neighbors + 1)
    rows   = np.repeat(np.arange(N), cfg.k_neighbors)
    cols   = nb[:, 1:].ravel()
    edge_index = torch.LongTensor(np.vstack([rows, cols]))
    print(f"[Graph] edges = {edge_index.shape[1]}")

    # Sobolev derivative grids
    du_exact_2d = np.gradient(exact, dx_val, axis=1).astype(np.float32)
    dt_exact_2d = np.gradient(exact, dt_val, axis=0).astype(np.float32)
    du_exact_grid = torch.from_numpy(du_exact_2d).to(device)
    dt_exact_grid = torch.from_numpy(dt_exact_2d).to(device)

    results, histories = [], []

    # ══════════════════════════════════════════
    # RUN 1 — Diffusion
    # ══════════════════════════════════════════
    m1, h1, pred1, w1, mse1, rl2_1 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="Diffusion-AC"
    )
    w11_1 = w1_1_rel_error(pred1, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_1:.4f}")
    results.append(("Diffusion-AC", mse1, rl2_1, w11_1))
    histories.append(h1)
    plot_solution(t_vec, x_vec, exact, pred1, rl2_1,
                  "Diffusion | Allen-Cahn", "results_ac/lqg_diff_solution.png")
    plot_holonomy_spectrum(w1, "Diffusion AC", "results_ac/holonomy_spectrum_diff.png")
    r_iface_1 = plot_holonomy_spatial(
        m1, X_np, edge_index, cfg,
        "Diffusion AC", "results_ac/holonomy_spatial_diff.png")

    # ══════════════════════════════════════════
    # RUN 2 — Diff+Sob  (main result)
    # ══════════════════════════════════════════
    m2, h2, pred2, w2, mse2, rl2_2 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="Diff+Sob-AC",
        Nt=Nt, Nx=Nx, dx_val=dx_val, dt_val=dt_val,
        du_exact_grid=du_exact_grid, dt_exact_grid=dt_exact_grid
    )
    w11_2 = w1_1_rel_error(pred2, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_2:.4f}")
    results.append(("Diff+Sob-AC", mse2, rl2_2, w11_2))
    histories.append(h2)
    plot_solution(t_vec, x_vec, exact, pred2, rl2_2,
                  "Diff+Sob | Allen-Cahn", "results_ac/lqg_sob_solution.png")
    plot_holonomy_spectrum(w2, "Diff+Sob AC", "results_ac/holonomy_spectrum_sob.png")
    r_iface_2 = plot_holonomy_spatial(
        m2, X_np, edge_index, cfg,
        "Diff+Sob AC", "results_ac/holonomy_spatial_sob.png")

    # ══════════════════════════════════════════
    # RUN 3 — MLP-Only  (Fourier baseline)
    # ══════════════════════════════════════════
    m3, h3, pred3, _, mse3, rl2_3 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=True, tag="MLP-Only-AC"
    )
    w11_3 = w1_1_rel_error(pred3, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_3:.4f}")
    results.append(("MLP-Only-AC", mse3, rl2_3, w11_3))
    histories.append(h3)
    plot_solution(t_vec, x_vec, exact, pred3, rl2_3,
                  "MLP-Only | Allen-Cahn", "results_ac/mlp_solution.png")

    # ══════════════════════════════════════════
    # RUN 4 — Fixed-Holonomy  (null ablation)
    # ══════════════════════════════════════════
    m4, h4, pred4, w4, mse4, rl2_4 = train_model(
        cfg, X_np, y_np, edge_index, device,
        ablate=False, tag="Fixed-Holonomy-AC", freeze_w=True
    )
    w11_4 = w1_1_rel_error(pred4, exact, x_vec, t_vec)
    print(f"  ✦ W1,1 : {w11_4:.4f}")
    results.append(("Fixed-Holonomy-AC", mse4, rl2_4, w11_4))
    histories.append(h4)
    plot_solution(t_vec, x_vec, exact, pred4, rl2_4,
                  "Fixed-Holonomy | Allen-Cahn", "results_ac/fixed_holo_solution.png")
    plot_holonomy_spectrum(w4, "Fixed (frozen) AC",
                           "results_ac/holonomy_spectrum_fixed.png")
    r_iface_4 = plot_holonomy_spatial(
        m4, X_np, edge_index, cfg,
        "Fixed-Holonomy AC", "results_ac/holonomy_spatial_fixed.png")

    # ══════════════════════════════════════════
    # Combined plots
    # ══════════════════════════════════════════
    all_preds  = [pred1, pred2, pred3, pred4]
    all_labels = ["Diff-AC","Diff+Sob-AC","MLP-Only-AC","Fixed-AC"]
    all_rl2s   = [rl2_1, rl2_2, rl2_3, rl2_4]

    plot_training_curves(histories, all_labels, "results_ac/training_curves.png")
    plot_interface_slices(t_vec, x_vec, exact, all_preds, all_labels, all_rl2s,
                          "results_ac/interface_slices.png")
    plot_metrics_bar(results, "results_ac/ablation_bar.png")

    summary_table(results)

    print(f"\n  R_interface summary:")
    print(f"    Diffusion  : {r_iface_1:.2f}×")
    print(f"    Diff+Sob   : {r_iface_2:.2f}×")
    print(f"    Fixed-Holonomy : {r_iface_4:.2f}×  (expect ≈ 1.00×)")
    print("\nAll results → results_ac/")


if __name__ == "__main__":
    main()