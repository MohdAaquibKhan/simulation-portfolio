"""
Surrogate Model (ROM) for FEA Parametric Studies — Demo
=========================================================
Demonstrates the surrogate/ROM methodology used to eliminate repeated solver runs
in parametric design studies.

In production (at BSH Hausgeräte GmbH):
  - Geometry and material parameters vary across hundreds of design points in Ansys Optislang
  - Full nonlinear FEA solver runs each point (~20–40 min per run)
  - A polynomial + RBF surrogate is trained on the DOE results
  - The surrogate replaces the solver for optimisation, enabling millisecond-scale inference

Here: a synthetic nonlinear response surface represents the FEA output.

Dependencies: numpy, scikit-learn, matplotlib
Run: python surrogate_model_demo.py

Author: Mohd Aaquib Khan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error


# ── 1. SYNTHETIC "FEA" RESPONSE SURFACE ──────────────────────────────────────
# Represents a nonlinear structural response (e.g. peak snap stress or deflection)
# as a function of two design variables: wall thickness t and fillet radius r

def fea_solver(t, r):
    """
    Synthetic nonlinear FEA response — mimics a snap-fit peak stress result.
    In production this is replaced by actual Ansys Mechanical solver execution.

    Parameters:
        t : wall thickness (mm), range [0.8, 2.5]
        r : fillet radius   (mm), range [0.2, 1.2]
    Returns:
        peak_stress (MPa)
    """
    # Nonlinear surface with interaction term
    stress = (200 / t**1.8) * (1 + 0.3 * np.sin(3 * t)) \
             - 40 * r + 15 * r**2 \
             + 8 * (t - 1.5) * (r - 0.7) \
             + np.random.normal(0, 2.5, size=np.shape(t))  # solver noise
    return np.clip(stress, 30, 450)


# ── 2. DOE SAMPLING (replaces Ansys Optislang DOE execution) ─────────────────

def run_doe(n_samples=80):
    """
    Latin Hypercube-style sampling over the design space.
    In production: Optislang generates the design points and runs the solver.
    """
    np.random.seed(7)
    t = np.random.uniform(0.8, 2.5, n_samples)
    r = np.random.uniform(0.2, 1.2, n_samples)
    s = fea_solver(t, r)
    return np.column_stack([t, r]), s


# ── 3. TRAIN SURROGATE MODELS ─────────────────────────────────────────────────

def train_surrogates(X_doe, y_doe):
    """
    Train two surrogate types:
    1. Polynomial regression (degree 3) — fast, interpretable
    2. Gaussian Process (RBF kernel) — captures complex nonlinearity, gives uncertainty

    Returns fitted models and cross-validation scores.
    """
    # ── Polynomial ──
    poly_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
        ("ridge",  Ridge(alpha=1.0))
    ])
    poly_cv = cross_val_score(poly_model, X_doe, y_doe, cv=5, scoring="r2")
    poly_model.fit(X_doe, y_doe)

    # ── Gaussian Process (RBF kernel) ──
    kernel   = RBF(length_scale=[0.5, 0.2]) + WhiteKernel(noise_level=5)
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                        normalize_y=True, random_state=42)
    gp_cv    = cross_val_score(gp_model, X_doe, y_doe, cv=5, scoring="r2")
    gp_model.fit(X_doe, y_doe)

    return poly_model, gp_model, poly_cv, gp_cv


# ── 4. VALIDATION ─────────────────────────────────────────────────────────────

def validate_surrogates(poly_model, gp_model, n_val=200):
    """
    Validate against unseen solver evaluations.
    In production: Optislang runs a dedicated validation set.
    """
    np.random.seed(99)
    t_val = np.random.uniform(0.8, 2.5, n_val)
    r_val = np.random.uniform(0.2, 1.2, n_val)
    X_val = np.column_stack([t_val, r_val])
    y_val = fea_solver(t_val, r_val)

    y_poly = poly_model.predict(X_val)
    y_gp, y_gp_std = gp_model.predict(X_val, return_std=True)

    results = {
        "poly": {"r2": r2_score(y_val, y_poly),
                 "mae": mean_absolute_error(y_val, y_poly)},
        "gp":   {"r2": r2_score(y_val, y_gp),
                 "mae": mean_absolute_error(y_val, y_gp),
                 "y_pred": y_gp, "y_true": y_val, "std": y_gp_std}
    }
    return results


# ── 5. DESIGN SPACE EXPLORATION (what the ROM enables) ───────────────────────

def explore_design_space(gp_model):
    """
    Once the ROM is trained, sweep the full design space at millisecond cost.
    In production: used for optimisation, sensitivity analysis, and design target checks.
    """
    t_grid = np.linspace(0.8, 2.5, 120)
    r_grid = np.linspace(0.2, 1.2, 100)
    T, R   = np.meshgrid(t_grid, r_grid)
    X_grid = np.column_stack([T.ravel(), R.ravel()])
    S_pred, S_std = gp_model.predict(X_grid, return_std=True)
    return T, R, S_pred.reshape(T.shape), S_std.reshape(T.shape)


# ── 6. VISUALISATION ──────────────────────────────────────────────────────────

def plot_results(X_doe, y_doe, val_results, T, R, S_pred, S_std):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Surrogate Model (ROM) for FEA Parametric Study — Demo",
                 fontsize=13, fontweight="bold", color="#1F3A5F")

    # ── A: DOE scatter ──
    ax1 = fig.add_subplot(2, 3, 1)
    sc = ax1.scatter(X_doe[:, 0], X_doe[:, 1], c=y_doe, cmap="RdYlGn_r",
                     s=40, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax1, label="Peak Stress (MPa)")
    ax1.set_title("DOE Sampling\n(solver evaluations)", fontweight="bold")
    ax1.set_xlabel("Wall Thickness t (mm)")
    ax1.set_ylabel("Fillet Radius r (mm)")

    # ── B: GP predicted surface ──
    ax2 = fig.add_subplot(2, 3, 2)
    cs  = ax2.contourf(T, R, S_pred, levels=20, cmap="RdYlGn_r")
    plt.colorbar(cs, ax=ax2, label="Predicted Stress (MPa)")
    ax2.set_title("GP Surrogate\nPredicted Stress Surface", fontweight="bold")
    ax2.set_xlabel("Wall Thickness t (mm)")
    ax2.set_ylabel("Fillet Radius r (mm)")

    # ── C: Uncertainty (std dev) ──
    ax3 = fig.add_subplot(2, 3, 3)
    cu  = ax3.contourf(T, R, S_std, levels=15, cmap="Blues")
    plt.colorbar(cu, ax=ax3, label="Prediction Std Dev (MPa)")
    ax3.set_title("GP Uncertainty\n(where to add more DOE points)", fontweight="bold")
    ax3.set_xlabel("Wall Thickness t (mm)")
    ax3.set_ylabel("Fillet Radius r (mm)")

    # ── D: Parity plot (predicted vs. actual) ──
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(val_results["gp"]["y_true"], val_results["gp"]["y_pred"],
                s=8, alpha=0.5, c="#2E609E")
    lims = [val_results["gp"]["y_true"].min(), val_results["gp"]["y_true"].max()]
    ax4.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax4.set_title("Validation Parity Plot\n(GP surrogate vs. solver)", fontweight="bold")
    ax4.set_xlabel("Solver Output (MPa)")
    ax4.set_ylabel("Surrogate Prediction (MPa)")
    ax4.legend(fontsize=8)
    ax4.text(0.05, 0.90,
             f"R² = {val_results['gp']['r2']:.4f}\n"
             f"MAE = {val_results['gp']['mae']:.2f} MPa",
             transform=ax4.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── E: Sensitivity — stress vs. thickness at fixed r ──
    ax5 = fig.add_subplot(2, 3, 5)
    t_sweep = np.linspace(0.8, 2.5, 200)
    for r_val, col in [(0.3, "#1F3A5F"), (0.7, "#2E609E"), (1.1, "#90B4D4")]:
        X_sw = np.column_stack([t_sweep, np.full_like(t_sweep, r_val)])
        s_sw, s_sw_std = gp_model.predict(X_sw, return_std=True)
        ax5.plot(t_sweep, s_sw, color=col, label=f"r = {r_val} mm")
        ax5.fill_between(t_sweep, s_sw - s_sw_std, s_sw + s_sw_std,
                         alpha=0.15, color=col)
    ax5.axhline(200, color="red", linestyle="--", lw=1.2, label="Design limit 200 MPa")
    ax5.set_title("Sensitivity Analysis\n(t sweep at fixed r)", fontweight="bold")
    ax5.set_xlabel("Wall Thickness t (mm)")
    ax5.set_ylabel("Peak Stress (MPa)")
    ax5.legend(fontsize=8)

    # ── F: Text summary ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary = (
        "SURROGATE ACCURACY\n"
        "─────────────────────────────\n\n"
        f"  Polynomial (deg-3)\n"
        f"  R²  = {val_results['poly']['r2']:.4f}\n"
        f"  MAE = {val_results['poly']['mae']:.2f} MPa\n\n"
        f"  Gaussian Process (RBF)\n"
        f"  R²  = {val_results['gp']['r2']:.4f}\n"
        f"  MAE = {val_results['gp']['mae']:.2f} MPa\n\n"
        "─────────────────────────────\n\n"
        "PRODUCTION IMPACT\n\n"
        "  DOE runs:   80 solver calls\n"
        "  Surrogate:  ∞ instant predictions\n"
        "  Speedup:    solver ~30 min each\n"
        "              surrogate < 1 ms\n\n"
        "  This enables full design-space\n"
        "  optimisation without a single\n"
        "  additional solver run."
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F0F4F8", alpha=0.8))

    plt.tight_layout()
    plt.savefig("surrogate_model_result.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved: surrogate_model_result.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running DOE (simulating solver evaluations) ...")
    X_doe, y_doe = run_doe(n_samples=80)
    print(f"  {len(X_doe)} design points evaluated.\n")

    print("Training surrogate models ...")
    poly_model, gp_model, poly_cv, gp_cv = train_surrogates(X_doe, y_doe)
    print(f"  Polynomial R² (5-fold CV): {poly_cv.mean():.4f} ± {poly_cv.std():.4f}")
    print(f"  GP R²          (5-fold CV): {gp_cv.mean():.4f} ± {gp_cv.std():.4f}\n")

    print("Validating against unseen solver evaluations ...")
    val_results = validate_surrogates(poly_model, gp_model)
    print(f"  Polynomial — R²: {val_results['poly']['r2']:.4f},  "
          f"MAE: {val_results['poly']['mae']:.2f} MPa")
    print(f"  GP          — R²: {val_results['gp']['r2']:.4f},  "
          f"MAE: {val_results['gp']['mae']:.2f} MPa\n")

    print("Sweeping full design space via surrogate ...")
    T, R, S_pred, S_std = explore_design_space(gp_model)

    plot_results(X_doe, y_doe, val_results, T, R, S_pred, S_std)
    print("\nDone.")
