import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import plot_config as cfg


def _auto_sparse_transformer_traj_npz() -> str:
    cands = sorted(
        list(cfg.RESULT_DIR.glob("trajectories_dataset_l96_*_sparse.npz")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    if not cands:
        raise FileNotFoundError(
            "Missing sparse transformer trajectories npz. Expected something like: result/trajectories_dataset_l96_*_sparse.npz"
        )
    return cands[-1].name


traj_file = cfg.RESULT_DIR / _auto_sparse_transformer_traj_npz()
data = np.load(traj_file, allow_pickle=True)

X_true = data["X_true"]
X_enkf = data["X_enkf"]
X_eakf = data["X_eakf"]
X_fused_enkf = data["X_fused_enkf"]
X_fused_eakf = data["X_fused_eakf"]
dt_obs = float(data["dt_obs"])

err_enkf = X_enkf - X_true
err_eakf = X_eakf - X_true
err_fused_enkf = X_fused_enkf - X_true
err_fused_eakf = X_fused_eakf - X_true

n_steps, n_vars = X_true.shape
time_max = n_steps * dt_obs

all_err = np.concatenate(
    [
        err_enkf.ravel(),
        err_fused_enkf.ravel(),
        err_eakf.ravel(),
        err_fused_eakf.ravel(),
    ]
)
vmax = float(np.percentile(np.abs(all_err), 85))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

cmap = LinearSegmentedColormap.from_list(
    "RedYellowGreen",
    ["#ff1744", "#ffea00", "#00c853"],
    N=256,
)

methods = [
    (err_enkf, "EnKF", "enkf"),
    (err_fused_enkf, "EnKF-TF", "enkf_tf"),
    (err_eakf, "EAKF", "eakf"),
    (err_fused_eakf, "EAKF-TF", "eakf_tf"),
]

fig, axes = plt.subplots(2, 2, figsize=cfg.get_figsize("2x2"), sharex=True, sharey=True)

plot_configs = [
    (0, 0, err_enkf, "EnKF"),
    (0, 1, err_fused_enkf, "EnKF-TF"),
    (1, 0, err_eakf, "EAKF"),
    (1, 1, err_fused_eakf, "EAKF-TF"),
]

for row, col, err, title in plot_configs:
    ax = axes[row, col]

    im = ax.imshow(err, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower", interpolation="bilinear",
                   extent=[0, n_vars, 0, time_max])

    ax.set_title(title, fontsize=cfg.FONT_SIZE["title"], fontweight='bold')

    if col == 0:
        ax.set_ylabel("Time", fontsize=cfg.FONT_SIZE["label"])
    if row == 1:
        ax.set_xlabel("State Variable Index", fontsize=cfg.FONT_SIZE["label"])

plt.tight_layout(rect=[0, 0, 0.88, 1])

cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Estimation Error", fontsize=cfg.FONT_SIZE["label"])

cfg.save_fig(fig, "fig9_hovmoller")
plt.show()
