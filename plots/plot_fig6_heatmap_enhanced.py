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

def load_trajectory_data():
    traj_file = cfg.RESULT_DIR / _auto_sparse_transformer_traj_npz()
    try:
        data = np.load(traj_file, allow_pickle=True)
        return {
            'X_true': data["X_true"],
            'X_enkf': data["X_enkf"],
            'X_eakf': data["X_eakf"],
            'X_fused_enkf': data["X_fused_enkf"],
            'X_fused_eakf': data["X_fused_eakf"],
            'dt_obs': float(data["dt_obs"])
        }
    except FileNotFoundError:
        return None

def plot_enhanced_heatmap():
    base_data = load_trajectory_data()
    if base_data is None:
        return

    X_true = base_data['X_true']
    dt_obs = base_data['dt_obs']

    errors = {}
    errors['enkf'] = base_data['X_enkf'] - X_true
    errors['eakf'] = base_data['X_eakf'] - X_true
    errors['enkf_tf'] = base_data['X_fused_enkf'] - X_true
    errors['eakf_tf'] = base_data['X_fused_eakf'] - X_true

    n_steps, n_vars = X_true.shape
    time_max = n_steps * dt_obs

    all_errors = np.concatenate([errors[key].flatten() for key in errors.keys()])
    vmax = np.percentile(np.abs(all_errors), 85)

    cmap = LinearSegmentedColormap.from_list(
        "RedYellowGreen",
        ["#ff1744", "#ffea00", "#00c853"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    methods = [
        ('EnKF', 'enkf'),
        ('EnKF-TF', 'enkf_tf'),
        ('EAKF', 'eakf'),
        ('EAKF-TF', 'eakf_tf')
    ]

    fig, axes = plt.subplots(2, 2, figsize=cfg.get_figsize("heatmap"), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')

    for i, (title, key) in enumerate(methods):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        im = ax.imshow(
            errors[key],
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=[0, n_vars, 0, time_max],
            interpolation="bilinear",
        )

        ax.set_title(title, fontsize=cfg.FONT_SIZE["title"], fontweight='bold', pad=10)

        if row == 1:
            ax.set_xlabel('State Variable Index', fontsize=cfg.FONT_SIZE["label"])
        if col == 0:
            ax.set_ylabel('Time', fontsize=cfg.FONT_SIZE["label"])

        ax.tick_params(labelsize=cfg.FONT_SIZE["tick"])

        ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
        ax.set_yticks([0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5])

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Estimation Error', fontsize=cfg.FONT_SIZE["label"], labelpad=15)
    cbar.ax.tick_params(labelsize=cfg.FONT_SIZE["tick"])

    cbar_ticks = np.linspace(-vmax, vmax, 9)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick:.0f}' for tick in cbar_ticks])

    cfg.save_fig(fig, "fig6_heatmap_enhanced")
    plt.show()

if __name__ == "__main__":
    plot_enhanced_heatmap()