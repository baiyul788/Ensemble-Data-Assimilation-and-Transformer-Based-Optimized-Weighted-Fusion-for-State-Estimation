import numpy as np
import matplotlib.pyplot as plt
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

def load_all_trajectories():
    trajectories = {}
    traj_file = cfg.RESULT_DIR / _auto_sparse_transformer_traj_npz()
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
    bilstm_file = cfg.RESULT_DIR / "trajectories_dataset_l96_n36_dt0p01_T20_dtobs0p2_s6_N10_sigObs2_sigModel0p1_seed0_sparse_bilstm.npz"
    try:
        bilstm_data = np.load(bilstm_file, allow_pickle=True)
        trajectories['bilstm'] = bilstm_data
    except FileNotFoundError:
        trajectories['bilstm'] = None
    return trajectories

def plot_enhanced_scatter():
    trajectories = load_all_trajectories()
    data = trajectories['transformer']
    X_true = data["X_true"].flatten()
    X_enkf = data["X_enkf"].flatten()
    X_eakf = data["X_eakf"].flatten()
    X_fused_enkf = data["X_fused_enkf"].flatten()
    X_fused_eakf = data["X_fused_eakf"].flatten()
    has_lstm = trajectories['lstm'] is not None
    if has_lstm:
        lstm_data = trajectories['lstm']
        X_fused_enkf_lstm = lstm_data["X_fused_enkf"].flatten()
        X_fused_eakf_lstm = lstm_data["X_fused_eakf"].flatten()
    else:
        X_fused_enkf_lstm = None
        X_fused_eakf_lstm = None
    has_bilstm = trajectories['bilstm'] is not None
    if has_bilstm:
        bilstm_data = trajectories['bilstm']
        X_fused_enkf_bilstm = bilstm_data["X_fused_enkf"].flatten()
        X_fused_eakf_bilstm = bilstm_data["X_fused_eakf"].flatten()
    else:
        X_fused_enkf_bilstm = None
        X_fused_eakf_bilstm = None
    fig, axes = plt.subplots(2, 4, figsize=cfg.get_figsize("2x4"))
    plot_configs = [
        ("EnKF", X_enkf, cfg.get_color("enkf"), (0, 0)),
        ("EnKF-TF", X_fused_enkf, cfg.get_color("fused_enkf"), (0, 1)),
        ("EnKF-LSTM", X_fused_enkf_lstm, '#DDA0DD', (0, 2)),
        ("EnKF-BiLSTM", X_fused_enkf_bilstm, '#FFB6C1', (0, 3)),
        ("EAKF", X_eakf, cfg.get_color("eakf"), (1, 0)),
        ("EAKF-TF", X_fused_eakf, cfg.get_color("fused_eakf"), (1, 1)),
        ("EAKF-LSTM", X_fused_eakf_lstm, '#F0E68C', (1, 2)),
        ("EAKF-BiLSTM", X_fused_eakf_bilstm, '#FFA07A', (1, 3)),
    ]
    vmin, vmax = min(X_true.min(), -12), max(X_true.max(), 15)
    for method_name, X_est, color, (row, col) in plot_configs:
        ax = axes[row, col]
        if X_est is None:
            ax.text(0.5, 0.5, f"{method_name}\nData Missing\nNeed Retraining", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=cfg.FONT_SIZE["title"], fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        else:
            ax.scatter(X_true, X_est, s=1.5, alpha=0.4, color=color, edgecolors="none")
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2, alpha=0.7)
            corr = np.corrcoef(X_true, X_est)[0, 1]
            rmse = np.sqrt(np.mean((X_true - X_est)**2))
            ax.text(0.05, 0.95, f"RMSE: {rmse:.3f}\nCorr: {corr:.3f}", 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9),
                   fontsize=cfg.FONT_SIZE["annotation"])
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        ax.set_title(method_name, fontsize=cfg.FONT_SIZE["title"], fontweight='bold', pad=10)
        ax.set_xlabel("True State", fontsize=cfg.FONT_SIZE["label"])
        ax.set_ylabel("Estimated State", fontsize=cfg.FONT_SIZE["label"])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=cfg.FONT_SIZE["tick"])
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.0)
    cfg.save_fig(fig, "fig8_scatter_enhanced")
    plt.show()

if __name__ == "__main__":
    plot_enhanced_scatter()