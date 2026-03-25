import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plot_config as cfg

df = pd.read_csv(cfg.RESULT_DIR / "sensitivity_summary.csv")

PARAM_ORDER = {"s": 0, "N": 1, "sigma_obs": 2, "sigma_model": 3}
df["param_order"] = df["param"].map(PARAM_ORDER)
df["value"] = pd.to_numeric(df["value"])
df = df.sort_values(["param_order", "value"]).reset_index(drop=True)

def make_label(row):
    param, val = row["param"], row["value"]
    if param == "s":
        return r"$S$ = " + f"{int(val)}"
    elif param == "N":
        return r"$N$ = " + f"{int(val)}"
    elif param == "sigma_obs":
        return r"$\sigma_{obs}$ = " + f"{val}"
    elif param == "sigma_model":
        return r"$\sigma_{model}$ = " + f"{val}"
    return f"{param} = {val}"

df["label"] = df.apply(make_label, axis=1)

all_rmse = pd.concat([df["rmse_enkf"], df["rmse_fused_enkf"], df["rmse_eakf"], df["rmse_fused_eakf"]])
x_min, x_max = max(2.5, all_rmse.min() - 0.5), all_rmse.max() + 0.5

fig, axes = plt.subplots(1, 2, figsize=cfg.get_figsize("double"), sharey=True)

configs = [
    ("EnKF", "rmse_enkf", "rmse_fused_enkf", cfg.get_color("enkf"), cfg.get_color("fused_enkf"), axes[0]),
    ("EAKF", "rmse_eakf", "rmse_fused_eakf", cfg.get_color("eakf"), cfg.get_color("fused_eakf"), axes[1]),
]

for title, col_da, col_fused, color, fused_color, ax in configs:
    y_pos = np.arange(len(df))
    
    for i, (_, row) in enumerate(df.iterrows()):
        da_val, fused_val = row[col_da], row[col_fused]
        ax.plot([fused_val, da_val], [i, i], color="gray", linewidth=1.5, alpha=0.7, zorder=1)
    
    ax.scatter(df[col_da], y_pos, color=color, s=60, zorder=2, edgecolors="black", linewidths=0.5)
    ax.scatter(df[col_fused], y_pos, color=fused_color, s=60, zorder=2, edgecolors="black", linewidths=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_xlabel("RMSE", fontsize=cfg.FONT_SIZE["label"])
    ax.set_title(f"{title} → {title}-TF", fontsize=cfg.FONT_SIZE["title"], fontweight='bold')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=cfg.GRID_ALPHA, axis="x")
    
    for y in [2.5, 5.5, 8.5]:
        ax.axhline(y, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

axes[0].set_yticklabels(df["label"], fontsize=cfg.FONT_SIZE["tick"])
axes[0].set_ylabel("Configuration", fontsize=cfg.FONT_SIZE["label"])

legend_handles = [
    Patch(facecolor=cfg.get_color("enkf"), edgecolor="black", label="EnKF"),
    Patch(facecolor=cfg.get_color("eakf"), edgecolor="black", label="EAKF"),
    Patch(facecolor=cfg.get_color("fused_enkf"), edgecolor="black", label="EnKF-TF"),
    Patch(facecolor=cfg.get_color("fused_eakf"), edgecolor="black", label="EAKF-TF"),
]
axes[1].legend(handles=legend_handles, loc="upper right", ncol=2, 
              fontsize=cfg.FONT_SIZE["legend"], framealpha=0.9)

plt.tight_layout()
cfg.save_fig(fig, "fig7_dumbbell")
plt.show()
