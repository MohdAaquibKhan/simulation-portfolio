"""
Physics-Informed Neural Network (PINN) — Damped Harmonic Oscillator
===================================================================
Demonstrates a PINN solving the canonical structural-dynamics equation:

        m x''(t) + c x'(t) + k x(t) = 0          (free damped vibration)

A PINN is a neural network x_NN(t) trained so that its OUTPUT obeys the
governing ODE. The physics enters the loss function directly via automatic
differentiation — no training labels and no mesh are required.

This script runs TWO experiments:

  1. FORWARD problem   — coefficients (m, c, k) are KNOWN. The PINN solves the
                          ODE from the physics + initial conditions alone, with
                          NO data. Validated against the exact analytical solution.

  2. INVERSE problem   — coefficients c and k are UNKNOWN. Given only a handful
                          of sparse, NOISY "sensor" measurements, the PINN
                          simultaneously (a) reconstructs the full motion and
                          (b) DISCOVERS c and k. This is ML-based system
                          identification / model correlation.

Outputs:
  pinn_forward_result.png      — forward solution, loss history, phase portrait
  pinn_inverse_result.png      — inverse reconstruction + parameter discovery
  pinn_training_evolution.gif   — animation of the forward solution converging

Dependencies: torch, numpy, matplotlib  (Pillow for the GIF)
Run: python pinn_damped_oscillator_demo.py

Author: Mohd Aaquib Khan
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
# True system: m x'' + c x' + k x = 0, with x(0) = X0, x'(0) = V0
M_TRUE, C_TRUE, K_TRUE = 1.0, 0.4, 4.0      # mass, damping, stiffness
X0, V0  = 1.0, 0.0                           # initial displacement, velocity
T_MAX   = 10.0                               # simulate 0 .. 10 s

# Derived modal quantities (for reference / labelling)
OMEGA0  = np.sqrt(K_TRUE / M_TRUE)           # natural frequency  = 2.0 rad/s
ZETA    = C_TRUE / (2 * np.sqrt(K_TRUE * M_TRUE))   # damping ratio = 0.1 (underdamped)


def exact_solution(t, m=M_TRUE, c=C_TRUE, k=K_TRUE, x0=X0, v0=V0):
    """Analytical solution of the underdamped free-vibration ODE."""
    w0   = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    wd   = w0 * np.sqrt(1 - zeta**2)          # damped natural frequency
    A    = x0
    B    = (v0 + zeta * w0 * x0) / wd
    return np.exp(-zeta * w0 * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Small fully-connected network x_NN(t).
    tanh activations — smooth and infinitely differentiable, which matters
    because we take 2nd derivatives of the output via autograd.
    Input t is normalised to [-1, 1] internally for stable training.
    """
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        net = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, t):
        t_norm = 2.0 * t / T_MAX - 1.0        # map [0, T_MAX] -> [-1, 1]
        return self.net(t_norm)


def derivatives(model, t):
    """Return x, x', x'' via automatic differentiation."""
    t = t.clone().requires_grad_(True)
    x   = model(t)
    dx  = torch.autograd.grad(x,  t, torch.ones_like(x),  create_graph=True)[0]
    ddx = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
    return x, dx, ddx


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 — FORWARD PROBLEM  (coefficients known, no data)
# ─────────────────────────────────────────────────────────────────────────────
def train_forward(adam_epochs=12000, lbfgs_iters=2500, n_collocation=300, capture_every=300):
    print("\n" + "=" * 64)
    print("  EXPERIMENT 1 — FORWARD PROBLEM (physics only, no data)")
    print("=" * 64)
    print(f"  Known coefficients:  m={M_TRUE}, c={C_TRUE}, k={K_TRUE}")
    print(f"  Natural frequency w0 = {OMEGA0:.3f} rad/s,  damping ratio = {ZETA:.3f}")

    model = PINN().to(DEVICE)

    # Collocation points where the ODE residual is enforced (mesh-free sampling)
    t_col = torch.linspace(0, T_MAX, n_collocation, device=DEVICE).reshape(-1, 1)
    # Single point for the initial conditions
    t_ic  = torch.zeros(1, 1, device=DEVICE)

    history = {"total": [], "phys": [], "ic": []}
    frames  = []                                    # snapshots for the GIF
    t_plot  = torch.linspace(0, T_MAX, 400, device=DEVICE).reshape(-1, 1)

    def compute_losses():
        x, dx, ddx = derivatives(model, t_col)
        residual   = M_TRUE * ddx + C_TRUE * dx + K_TRUE * x
        loss_phys  = torch.mean(residual**2)
        x0, dx0, _ = derivatives(model, t_ic)
        loss_ic    = ((x0 - X0)**2 + (dx0 - V0)**2).squeeze()
        return loss_phys, loss_ic

    def log(ep, loss, lp, li):
        history["total"].append(loss); history["phys"].append(lp); history["ic"].append(li)

    # ── Phase 1: Adam (fast global search) ──────────────────────────────────
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.5)
    for ep in range(adam_epochs):
        opt.zero_grad()
        lp, li = compute_losses()
        loss = lp + 100.0 * li                      # weight ICs higher
        loss.backward(); opt.step(); sched.step()
        log(ep, loss.item(), lp.item(), li.item())
        if ep % capture_every == 0:
            with torch.no_grad():
                frames.append((ep, model(t_plot).cpu().numpy().flatten()))
        if ep % 2000 == 0:
            print(f"  [Adam]  epoch {ep:6d} | total {loss.item():.3e} | "
                  f"phys {lp.item():.3e} | ic {li.item():.3e}")

    # ── Phase 2: LBFGS (second-order polish — kills spectral-bias error) ────
    print("  [LBFGS] refining ...")
    opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters,
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-9, tolerance_change=1e-12)
    lbfgs_state = {"n": 0}

    def closure():
        opt2.zero_grad()
        lp, li = compute_losses()
        loss = lp + 100.0 * li
        loss.backward()
        log(adam_epochs + lbfgs_state["n"], loss.item(), lp.item(), li.item())
        if lbfgs_state["n"] % 50 == 0:
            with torch.no_grad():
                frames.append((adam_epochs + lbfgs_state["n"],
                               model(t_plot).cpu().numpy().flatten()))
        lbfgs_state["n"] += 1
        return loss

    opt2.step(closure)

    # Final frame
    with torch.no_grad():
        t_np   = t_plot.cpu().numpy().flatten()
        x_pinn = model(t_plot).cpu().numpy().flatten()
    frames.append((adam_epochs + lbfgs_state["n"], x_pinn))

    x_exact = exact_solution(t_np)
    l2  = np.sqrt(np.mean((x_pinn - x_exact)**2))
    rel = l2 / np.sqrt(np.mean(x_exact**2)) * 100
    print(f"  [LBFGS] final total loss: {history['total'][-1]:.3e}")
    print(f"\n  Final L2 error vs analytical: {l2:.4e}  ({rel:.3f}% relative)")

    return model, history, frames, t_np, x_pinn, x_exact


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — INVERSE PROBLEM  (discover c and k from sparse noisy data)
# ─────────────────────────────────────────────────────────────────────────────
def train_inverse(adam_epochs=15000, lbfgs_iters=1500, n_data=20, noise=0.02, n_collocation=300):
    print("\n" + "=" * 64)
    print("  EXPERIMENT 2 — INVERSE PROBLEM (discover c, k from sparse data)")
    print("=" * 64)
    print(f"  TRUE (hidden) coefficients:  c={C_TRUE}, k={K_TRUE}")
    print(f"  Sensor data: {n_data} noisy points ({noise*100:.0f}% noise)")

    model = PINN().to(DEVICE)

    # Unknown coefficients become LEARNABLE parameters.
    # Start from deliberately wrong guesses to prove they are discovered.
    log_c = nn.Parameter(torch.tensor(np.log(1.5), dtype=torch.float32))  # guess c=1.5
    log_k = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))  # guess k=1.0
    # log-parameterisation keeps c, k strictly positive

    # Sparse, noisy "sensor" measurements from the true system
    t_data_np = np.sort(np.random.uniform(0, T_MAX, n_data))
    x_data_np = exact_solution(t_data_np) + noise * np.random.randn(n_data)
    t_data = torch.tensor(t_data_np, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    x_data = torch.tensor(x_data_np, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    t_col = torch.linspace(0, T_MAX, n_collocation, device=DEVICE).reshape(-1, 1)

    c_history, k_history, loss_history = [], [], []

    def compute_loss():
        c_val = torch.exp(log_c); k_val = torch.exp(log_k)
        x, dx, ddx = derivatives(model, t_col)
        residual   = M_TRUE * ddx + c_val * dx + k_val * x
        loss_phys  = torch.mean(residual**2)
        loss_dat   = torch.mean((model(t_data) - x_data)**2)
        return loss_phys + 10.0 * loss_dat, c_val, k_val

    # ── Phase 1: Adam ───────────────────────────────────────────────────────
    opt = torch.optim.Adam(list(model.parameters()) + [log_c, log_k], lr=3e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
    for ep in range(adam_epochs):
        opt.zero_grad()
        loss, c_val, k_val = compute_loss()
        loss.backward(); opt.step(); sched.step()
        c_history.append(c_val.item()); k_history.append(k_val.item())
        loss_history.append(loss.item())
        if ep % 2500 == 0:
            print(f"  [Adam]  epoch {ep:6d} | loss {loss.item():.3e} | "
                  f"c_est {c_val.item():.4f} (true {C_TRUE}) | "
                  f"k_est {k_val.item():.4f} (true {K_TRUE})")

    # ── Phase 2: LBFGS (polish both the field and the discovered coefficients) ──
    print("  [LBFGS] refining ...")
    opt2 = torch.optim.LBFGS(list(model.parameters()) + [log_c, log_k], lr=1.0,
                             max_iter=lbfgs_iters, history_size=50,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt2.zero_grad()
        loss, c_val, k_val = compute_loss()
        loss.backward()
        c_history.append(c_val.item()); k_history.append(k_val.item())
        loss_history.append(loss.item())
        return loss

    opt2.step(closure)

    c_final, k_final = np.exp(log_c.item()), np.exp(log_k.item())
    print(f"  [LBFGS] final loss: {loss_history[-1]:.3e}")
    print(f"\n  DISCOVERED:  c = {c_final:.4f}  (true {C_TRUE},  error {abs(c_final-C_TRUE)/C_TRUE*100:.2f}%)")
    print(f"  DISCOVERED:  k = {k_final:.4f}  (true {K_TRUE},  error {abs(k_final-K_TRUE)/K_TRUE*100:.2f}%)")

    with torch.no_grad():
        t_np   = np.linspace(0, T_MAX, 400)
        t_t    = torch.tensor(t_np, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        x_pinn = model(t_t).cpu().numpy().flatten()
    x_exact = exact_solution(t_np)

    return (dict(c_hist=c_history, k_hist=k_history, loss=loss_history,
                 t_data=t_data_np, x_data=x_data_np, t=t_np,
                 x_pinn=x_pinn, x_exact=x_exact, c_final=c_final, k_final=k_final))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
NAVY, BLUE, RED, GREEN = "#1F3A5F", "#2E609E", "#C0392B", "#27AE60"


def plot_forward(history, t, x_pinn, x_exact, model):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("PINN — Forward Problem: Damped Oscillator  (m x'' + c x' + k x = 0)",
                 fontsize=13, fontweight="bold", color=NAVY)

    # Solution overlay
    ax = axes[0]
    ax.plot(t, x_exact, color=RED, lw=3, alpha=0.5, label="Analytical (exact)")
    ax.plot(t, x_pinn, "--", color=NAVY, lw=1.8, label="PINN (physics only)")
    ax.set_title("Solution: PINN vs. Analytical", fontweight="bold")
    ax.set_xlabel("Time t (s)"); ax.set_ylabel("Displacement x(t)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.text(0.97, 0.95, "No training data used —\nlearned from the ODE alone",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#EAF2D3", alpha=0.8))

    # Loss history
    ax = axes[1]
    ax.semilogy(history["total"], color=NAVY, label="Total loss")
    ax.semilogy(history["phys"],  color=BLUE, alpha=0.7, label="Physics residual")
    ax.semilogy(history["ic"],    color=GREEN, alpha=0.7, label="Initial conditions")
    ax.set_title("Training Loss (log scale)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # Phase portrait
    ax = axes[2]
    t_t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    _, dx, _ = derivatives(model, t_t)
    dx = dx.detach().numpy().flatten()
    w0 = OMEGA0; zeta = ZETA; wd = w0*np.sqrt(1-zeta**2)
    A = X0; B = (V0 + zeta*w0*X0)/wd
    dx_exact = (np.exp(-zeta*w0*t) *
                (-zeta*w0*(A*np.cos(wd*t)+B*np.sin(wd*t))
                 + (-A*wd*np.sin(wd*t)+B*wd*np.cos(wd*t))))
    ax.plot(x_exact, dx_exact, color=RED, lw=3, alpha=0.5, label="Analytical")
    ax.plot(x_pinn, dx, "--", color=NAVY, lw=1.6, label="PINN")
    ax.set_title("Phase Portrait (x vs. dx/dt)", fontweight="bold")
    ax.set_xlabel("Displacement x"); ax.set_ylabel("Velocity dx/dt")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("pinn_forward_result.png", dpi=150, bbox_inches="tight")
    print("  Saved: pinn_forward_result.png")
    plt.close()


def plot_inverse(R):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("PINN — Inverse Problem: Discovering c and k from Sparse Noisy Data",
                 fontsize=13, fontweight="bold", color=NAVY)

    # Reconstruction + data
    ax = axes[0]
    ax.plot(R["t"], R["x_exact"], color=RED, lw=3, alpha=0.4, label="True system")
    ax.plot(R["t"], R["x_pinn"], "--", color=NAVY, lw=1.8, label="PINN reconstruction")
    ax.scatter(R["t_data"], R["x_data"], color=GREEN, s=55, zorder=5,
               edgecolors="k", linewidths=0.6, label="Noisy sensor data")
    ax.set_title("Reconstruction from sparse data", fontweight="bold")
    ax.set_xlabel("Time t (s)"); ax.set_ylabel("Displacement x(t)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Parameter convergence
    ax = axes[1]
    ax.plot(R["c_hist"], color=BLUE, label=f"c estimate -> {R['c_final']:.3f}")
    ax.plot(R["k_hist"], color=NAVY, label=f"k estimate -> {R['k_final']:.3f}")
    ax.axhline(C_TRUE, color=BLUE, ls="--", alpha=0.6, label=f"c true = {C_TRUE}")
    ax.axhline(K_TRUE, color=NAVY, ls="--", alpha=0.6, label=f"k true = {K_TRUE}")
    ax.set_title("Coefficient Discovery", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Estimated value")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    # Loss
    ax = axes[2]
    ax.semilogy(R["loss"], color=NAVY)
    ax.set_title("Training Loss (log scale)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
    ax.grid(alpha=0.3, which="both")
    ax.text(0.95, 0.92,
            f"Started:  c=1.50, k=1.00\n"
            f"Found:    c={R['c_final']:.3f}, k={R['k_final']:.3f}\n"
            f"True:     c={C_TRUE}, k={K_TRUE}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="#F0F4F8", alpha=0.9))

    plt.tight_layout()
    plt.savefig("pinn_inverse_result.png", dpi=150, bbox_inches="tight")
    print("  Saved: pinn_inverse_result.png")
    plt.close()


def animate_training(frames, t, x_exact):
    """GIF of the forward solution converging over training epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, x_exact, color=RED, lw=3, alpha=0.4, label="Analytical (target)")
    line, = ax.plot([], [], "--", color=NAVY, lw=2, label="PINN")
    txt = ax.text(0.97, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax.set_xlim(0, T_MAX)
    ymax = np.max(np.abs(x_exact)) * 1.3
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Time t (s)"); ax.set_ylabel("Displacement x(t)")
    ax.set_title("PINN learning the dynamics from the ODE", fontweight="bold", color=NAVY)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    def update(i):
        ep, x_pred = frames[i]
        line.set_data(t, x_pred)
        txt.set_text(f"epoch {ep}")
        return line, txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=True)
    anim.save("pinn_training_evolution.gif", writer=PillowWriter(fps=8))
    print("  Saved: pinn_training_evolution.gif")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Experiment 1 — forward
    model, hist, frames, t, x_pinn, x_exact = train_forward()
    print("\n  Rendering forward-problem figures ...")
    plot_forward(hist, t, x_pinn, x_exact, model)
    animate_training(frames, t, x_exact)

    # Experiment 2 — inverse
    R = train_inverse()
    print("\n  Rendering inverse-problem figures ...")
    plot_inverse(R)

    print("\nDone — 3 output files generated.")
