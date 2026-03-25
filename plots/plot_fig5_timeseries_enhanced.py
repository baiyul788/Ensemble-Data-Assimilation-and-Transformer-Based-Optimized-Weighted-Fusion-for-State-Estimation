import numpy as np
import matplotlib.pyplot as plt
import plot_config as cfg


def _auto_sparse_trajectory_npz() -> str:
    cands = sorted(
        list(cfg.RESULT_DIR.glob("trajectories_dataset_l96_*_sparse.npz")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    if not cands:
        raise FileNotFoundError(
            "Missing sparse trajectories npz. Expected something like: result/trajectories_dataset_l96_*_sparse.npz"
        )
    return cands[-1].name


def load_trajectory_data():
    trajectories = {}

    traj_file = cfg.RESULT_DIR / _auto_sparse_trajectory_npz()
    if not traj_file.exists():
        raise FileNotFoundError(f"Missing required trajectories file: {traj_file}")
    data = np.load(traj_file, allow_pickle=True)
    trajectories['transformer'] = data

    lstm_file = cfg.RESULT_DIR / "trajectories_dataset_l96_n36_dt0p01_T20_dtobs0p2_s6_N10_sigObs2_sigModel0p1_seed0_sparse_lstm.npz"
    try:
        lstm_data = np.load(lstm_file, allow_pickle=True)
        trajectories['lstm'] = lstm_data
    except FileNotFoundError:
        trajectories['lstm'] = None

    bilstm_files = [
        cfg.RESULT_DIR / "trajectories_bilstm_dataset_l96_n36_dt0p01_T20_dtobs0p2_s6_N10_sigObs2_sigModel0p1_seed0_sparse.npz",
        cfg.RESULT_DIR / "trajectories_dataset_l96_n36_dt0p01_T20_dtobs0p2_s6_N10_sigObs2_sigModel0p1_seed0_sparse_bilstm.npz",
        cfg.RESULT_DIR / "trajectories_bilstm_sparse.npz"
    ]

    trajectories['bilstm'] = None

    for bilstm_file in bilstm_files:
        try:
            bilstm_data = np.load(bilstm_file, allow_pickle=True)
            trajectories['bilstm'] = bilstm_data
            break
        except FileNotFoundError:
            continue

    return trajectories


def plot_enhanced_timeseries():
    trajectories = load_trajectory_data()

    base_data = trajectories['transformer']
    X_true = base_data["X_true"]
    X_enkf = base_data["X_enkf"]
    X_eakf = base_data["X_eakf"]
    X_fused_enkf = base_data["X_fused_enkf"]
    X_fused_eakf = base_data["X_fused_eakf"]
    dt_obs = float(base_data["dt_obs"])
    obs_indices = base_data["obs_indices"]

    n_steps = X_true.shape[0]
    time = np.arange(n_steps) * dt_obs

    var_obs = obs_indices[0]
    var_unobs = obs_indices[0] + 1

    t_start, t_end = 5, 15
    start_idx = max(0, int(t_start / dt_obs))
    end_idx = min(n_steps, int(t_end / dt_obs))

    if start_idx >= end_idx or end_idx > n_steps:
        start_idx = max(0, n_steps // 4)
        end_idx = min(n_steps, 3 * n_steps // 4)

    time_slice = slice(start_idx, end_idx)

    fig, axes = plt.subplots(2, 2, figsize=cfg.get_figsize("timeseries"))

    plot_configs = [
        (0, 0, var_obs, f"(a) Observed Variable ($x_{{{var_obs+1}}}$)"),
        (0, 1, var_unobs, f"(b) Unobserved Variable ($x_{{{var_unobs+1}}}$)"),
        (1, 0, var_obs, f"(c) Observed Variable - Fusion Comparison"),
        (1, 1, var_unobs, f"(d) Unobserved Variable - Fusion Comparison"),
    ]

    for row, col, var_idx, title in plot_configs:
        ax = axes[row, col]

        ax.plot(time[time_slice], X_true[time_slice, var_idx], 'k-', linewidth=2, label='Truth', alpha=0.8)

        if row == 0:
            ax.plot(time[time_slice], X_enkf[time_slice, var_idx], color=cfg.get_color("enkf"), linewidth=1.5, label='EnKF', alpha=0.7)
            ax.plot(time[time_slice], X_eakf[time_slice, var_idx], color=cfg.get_color("eakf"), linewidth=1.5, label='EAKF', alpha=0.7)
            ax.plot(time[time_slice], X_fused_eakf[time_slice, var_idx], color='#2E8B57', linewidth=1.5, label='Neural Networks', alpha=0.7, linestyle='--')
        else:
            ax.plot(time[time_slice], X_fused_enkf[time_slice, var_idx], color=cfg.get_color("fused_enkf"), linewidth=1.5, label='EnKF-TF', alpha=0.7)
            ax.plot(time[time_slice], X_fused_eakf[time_slice, var_idx], color=cfg.get_color("fused_eakf"), linewidth=1.5, label='EAKF-TF', alpha=0.7)
            ax.plot(time[time_slice], X_enkf[time_slice, var_idx], color=cfg.get_color("enkf"), linewidth=1, label='EnKF', alpha=0.5, linestyle=':')
            ax.plot(time[time_slice], X_eakf[time_slice, var_idx], color=cfg.get_color("eakf"), linewidth=1, label='EAKF', alpha=0.5, linestyle=':')

        ax.set_title(title, fontsize=cfg.FONT_SIZE["title"], fontweight='bold')
        ax.set_xlabel("Time", fontsize=cfg.FONT_SIZE["label"])
        ax.set_ylabel("State Value", fontsize=cfg.FONT_SIZE["label"])
        ax.grid(True, alpha=cfg.GRID_ALPHA)
        ax.legend(fontsize=cfg.FONT_SIZE["legend"], loc='upper right')

        ax.relim()
        ax.autoscale_view()

    plt.tight_layout()
    cfg.save_fig(fig, "fig5_timeseries_enhanced")
    plt.show()


if __name__ == "__main__":
    plot_enhanced_timeseries()