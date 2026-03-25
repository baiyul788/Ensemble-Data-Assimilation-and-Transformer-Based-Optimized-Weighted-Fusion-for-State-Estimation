import pandas as pd
import matplotlib.pyplot as plt
import plot_config as cfg

df = pd.read_csv(cfg.RESULT_DIR / "sensitivity_summary.csv")

PARAMS = [
    ("s", ["6", "9", "18"], r"Observation Dimension ($S$)"),
    ("N", ["10", "20", "40"], r"Ensemble Size ($N$)"),
    ("sigma_obs", ["1.0", "2.0", "4.0"], r"Observation Noise ($\sigma_{obs}$)"),
    ("sigma_model", ["0.0", "0.1", "0.2"], r"Model Error ($\sigma_{model}$)"),
]

fig, axes = plt.subplots(2, 2, figsize=cfg.get_figsize("2x2"))
axes = axes.flatten()

for i, (ax, (param, x_labels, xlabel)) in enumerate(zip(axes, PARAMS)):
    sub = df[df["param"] == param].copy()
    sub["value"] = pd.to_numeric(sub["value"])
    sub = sub.sort_values("value")
    
    x_pos = range(len(x_labels))
    
    ax.plot(x_pos, sub["improve_enkf"].values, "o-", 
            label="EnKF-TF vs EnKF", color=cfg.get_color("enkf"))
    ax.plot(x_pos, sub["improve_eakf"].values, "s--", 
            label="EAKF-TF vs EAKF", color=cfg.get_color("eakf"))
    
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(xlabel)
    if i % 2 == 0:
        ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-5, 50)
    ax.grid(True, alpha=cfg.GRID_ALPHA)
    
    ax.text(0.95, 0.95, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=cfg.FONT_SIZE["label"], fontweight="bold", va="top", ha="right")
    
    ax.legend(loc="upper right", fontsize=cfg.FONT_SIZE["legend"], framealpha=0.9)

plt.tight_layout()
cfg.save_fig(fig, "fig3_sensitivity")
plt.show()
