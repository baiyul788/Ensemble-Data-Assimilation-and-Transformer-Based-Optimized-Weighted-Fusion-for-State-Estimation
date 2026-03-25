import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch
import plot_config as cfg

def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")

WINDOW = 5
ALPHA_ENKF = 0.20
ALPHA_EAKF = 0.25

def load_seed_improvements():
    frames: list[pd.DataFrame] = []

    preferred_pattern = (
        f"sensitivity_sens_*_w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}.csv"
    )
    csv_paths = list(cfg.RESULT_DIR.glob(preferred_pattern))
    if not csv_paths:
        csv_paths = list(cfg.RESULT_DIR.glob("sensitivity_sens_*.csv"))

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        stem = csv_path.stem
        m = re.match(r"^sensitivity_(?P<tag>.+?)_w", stem)
        if not m:
            continue
        tag = m.group("tag")

        if tag.startswith("sens_sigma_obs"):
            param = "sigma_obs"
            val_str = tag[len("sens_sigma_obs") :]
        elif tag.startswith("sens_sigma_model"):
            param = "sigma_model"
            val_str = tag[len("sens_sigma_model") :]
        elif tag.startswith("sens_N"):
            param = "N"
            val_str = tag[len("sens_N") :]
        elif tag.startswith("sens_s"):
            param = "s"
            val_str = tag[len("sens_s") :]
        else:
            continue

        val_str = str(val_str).replace("p", ".").replace("m", "-")
        try:
            value = int(float(val_str)) if param in ("s", "N") else float(val_str)
        except ValueError:
            value = float("nan")

        if "seed" not in df.columns:
            continue

        df = df.copy()
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df = df[df["seed"].notna()]

        for c in ["rmse_enkf", "rmse_eakf", "rmse_fused_enkf", "rmse_fused_eakf"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["improve_enkf"] = np.where(
            df["rmse_enkf"] > 0,
            (df["rmse_enkf"] - df["rmse_fused_enkf"]) / df["rmse_enkf"] * 100,
            np.nan,
        )
        df["improve_eakf"] = np.where(
            df["rmse_eakf"] > 0,
            (df["rmse_eakf"] - df["rmse_fused_eakf"]) / df["rmse_eakf"] * 100,
            np.nan,
        )

        df["source"] = csv_path.stem
        df["param"] = param
        df["value"] = value
        frames.append(df[["seed", "improve_enkf", "improve_eakf", "source", "param", "value"]])

    if not frames:
        return pd.DataFrame(columns=["seed", "improve_enkf", "improve_eakf", "source", "param", "value"])

    out = pd.concat(frames, ignore_index=True)
    out = out[np.isfinite(out["improve_enkf"]) | np.isfinite(out["improve_eakf"])]
    return out

df = load_seed_improvements()

if df.empty:
    exit(1)

PARAM_ORDER = ["s", "N", "sigma_obs", "sigma_model"]
PARAM_TICKS = {
    "s": r"Observation Dimension" + "\n" + r"($S$)",
    "N": r"Ensemble Size" + "\n" + r"($N$)",
    "sigma_obs": r"Observation Noise" + "\n" + r"($\sigma_{obs}$)",
    "sigma_model": r"Model Error" + "\n" + r"($\sigma_{model}$)",
}

fig, ax = plt.subplots(figsize=cfg.get_figsize("double"))

SHOW_POINTS = False

data_series: list[np.ndarray] = []
positions: list[float] = []
group_centers: list[float] = []
group_lefts: list[float] = []
group_rights: list[float] = []

# layout: 4 groups, each with 2 boxes (EnKF/EAKF)
group_gap = 1.0
box_sep = 0.45

for g, param in enumerate(PARAM_ORDER):
    sub = df[df["param"] == param]
    d1 = sub["improve_enkf"].dropna().to_numpy(dtype=float)
    d2 = sub["improve_eakf"].dropna().to_numpy(dtype=float)

    base = g * (2 * box_sep + group_gap)
    p1 = base + 1.0
    p2 = base + 1.0 + box_sep

    data_series.extend([d1, d2])
    positions.extend([p1, p2])
    group_centers.append((p1 + p2) / 2.0)
    group_lefts.append(p1)
    group_rights.append(p2)

bp = ax.boxplot(
    data_series,
    positions=positions,
    widths=0.25,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "black", "linewidth": 1.2},
    whiskerprops={"color": "black", "linewidth": 1.0},
    capprops={"color": "black", "linewidth": 1.0},
)

colors = [cfg.get_color("enkf"), cfg.get_color("eakf")] * len(PARAM_ORDER)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")
    patch.set_linewidth(1.0)

ax.set_xticks(group_centers)
ax.set_xticklabels([PARAM_TICKS[p] for p in PARAM_ORDER])

for g in range(len(group_centers) - 1):
    boundary = (group_rights[g] + group_lefts[g + 1]) / 2.0
    ax.axvline(boundary, color=cfg.get_color("gray"), linewidth=0.6, alpha=0.35)

legend_handles = [
    Patch(facecolor=cfg.get_color("enkf"), edgecolor="black", alpha=0.7, label="EnKF-TF vs EnKF"),
    Patch(facecolor=cfg.get_color("eakf"), edgecolor="black", alpha=0.7, label="EAKF-TF vs EAKF"),
]
ax.legend(handles=legend_handles, loc="upper right", ncol=1, frameon=True, 
          fancybox=False, edgecolor="gray", framealpha=0.9)

ax.axhline(0, color="gray", linestyle="--", linewidth=1)

if SHOW_POINTS:
    _rng = np.random.default_rng(0)
    for pos, data in zip(positions, data_series):
        if len(data) == 0:
            continue
        x = _rng.normal(pos, 0.03, size=len(data))
        ax.scatter(x, data, alpha=0.18, s=10, color=cfg.get_color("gray"), zorder=2, linewidths=0)

ax.set_ylabel("Improvement (%)", fontsize=cfg.FONT_SIZE["label"])
ax.grid(True, alpha=0.18, axis="y")

all_vals = np.concatenate([d for d in data_series if len(d) > 0]) if any(len(d) > 0 for d in data_series) else np.array([])
if len(all_vals) > 0:
    y_min = min(float(np.nanmin(all_vals)), -5)
    y_max = max(float(np.nanmax(all_vals)), 55)
    pad = 2.0
    ax.set_ylim(y_min - pad, y_max + pad)

plt.tight_layout()
cfg.save_fig(fig, "fig4_boxplot")
plt.show()
