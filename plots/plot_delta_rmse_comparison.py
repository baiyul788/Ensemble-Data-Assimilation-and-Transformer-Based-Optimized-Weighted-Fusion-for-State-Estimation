import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plot_config as cfg


def _require_single_match(paths: list[Path], *, what: str) -> Path:
    paths = [Path(p) for p in paths if p is not None]
    if len(paths) == 1:
        return paths[0]
    if len(paths) == 0:
        raise FileNotFoundError(f"Missing required file for {what}.")
    # Deterministic: pick newest by mtime, but still error so user is aware.
    newest = sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)[-1]
    raise FileExistsError(
        f"Multiple candidates found for {what}: {[p.name for p in paths]}. "
        f"Please delete extras or pass --sparse_tf_csv explicitly. Newest: {newest.name}"
    )


def _auto_sparse_tf_csv() -> str:
    # Prefer full-config Plan A CSVs that embed localization/inflation.
    cands = sorted(
        list(cfg.RESULT_DIR.glob("plan_a_multiseed_*_loc*_infl*.csv")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    # Require at least one; do not silently fall back to legacy names.
    path = _require_single_match(cands[-1:] if len(cands) >= 1 else [], what="sparse TF CSV (plan_a_multiseed_*_loc*_infl*.csv)")
    return path.name

def _read_seed_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    seed_numeric = pd.to_numeric(df.get("seed"), errors="coerce")
    df = df[seed_numeric.notna()].copy()
    df["seed"] = seed_numeric[seed_numeric.notna()].astype(int)
    return df


def _delta_stats(df: pd.DataFrame) -> dict:
    # delta > 0 means improvement over DA baseline
    d_en = pd.to_numeric(df["rmse_enkf"], errors="coerce") - pd.to_numeric(df["rmse_fused_enkf"], errors="coerce")
    d_ea = pd.to_numeric(df["rmse_eakf"], errors="coerce") - pd.to_numeric(df["rmse_fused_eakf"], errors="coerce")

    def stats(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        if a.size == 1:
            return float(a.mean()), 0.0
        return float(a.mean()), float(a.std(ddof=1))

    men, sen = stats(d_en.values)
    mea, sea = stats(d_ea.values)
    return {
        "men": men,
        "sen": sen,
        "mea": mea,
        "sea": sea,
    }


def _fmt_val(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    # Avoid showing -0.000
    if abs(float(x)) < 5e-4:
        x = 0.0
    xf = float(x)
    if xf > 0:
        return f"+{xf:.3f}"
    return f"{xf:.3f}"


def _crosses_zero(mean: float, std: float) -> bool:
    if not np.isfinite(mean) or not np.isfinite(std):
        return False
    return (mean - std) <= 0.0 <= (mean + std)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_tf_csv", type=str, default="multiseed_summary_baseline_w5_aEn0p33_aEa0p39.csv")
    p.add_argument("--baseline_lstm_csv", type=str, default="multiseed_summary_lstm_baseline_w5_aEn0p33_aEa0p39.csv")
    p.add_argument("--baseline_bilstm_csv", type=str, default="multiseed_summary_bilstm_baseline_w5_aEn0p33_aEa0p39.csv")
    p.add_argument("--sparse_tf_csv", type=str, default="AUTO")
    p.add_argument("--sparse_lstm_csv", type=str, default="multiseed_summary_lstm_w5_aEn0p2_aEa0p25.csv")
    p.add_argument("--sparse_bilstm_csv", type=str, default="multiseed_summary_bilstm_w5_aEn0p2_aEa0p25.csv")
    p.add_argument("--out_name", type=str, default="delta_rmse_comparison")
    return p.parse_args()


def plot_delta_rmse_comparison():
    a = _parse_args()

    if str(a.sparse_tf_csv).strip().upper() == "AUTO":
        a.sparse_tf_csv = _auto_sparse_tf_csv()

    cfg.apply_style()

    specs = {
        "baseline": {
            "title": "Baseline Configuration",
            "tf": cfg.RESULT_DIR / a.baseline_tf_csv,
            "lstm": cfg.RESULT_DIR / a.baseline_lstm_csv,
            "bilstm": cfg.RESULT_DIR / a.baseline_bilstm_csv,
        },
        "sparse": {
            "title": "Sparse Configuration",
            "tf": cfg.RESULT_DIR / a.sparse_tf_csv,
            "lstm": cfg.RESULT_DIR / a.sparse_lstm_csv,
            "bilstm": cfg.RESULT_DIR / a.sparse_bilstm_csv,
        },
    }

    methods = [
        ("TF", "tf", "#90EE90", "#87CEEB"),
        ("LSTM", "lstm", "#DDA0DD", "#F0E68C"),
        ("BiLSTM", "bilstm", "#FFB6C1", "#FFA07A"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.get_figsize("double"), sharey=True)

    # Pre-compute global y-limits for fair comparison across subplots
    all_means = []
    all_stds = []
    for sp in specs.values():
        for _, key, _, _ in methods:
            path = sp[key]
            if not path.exists():
                continue
            df = _read_seed_rows(path)
            if df.empty:
                continue
            st = _delta_stats(df)
            all_means.extend([st["men"], st["mea"]])
            all_stds.extend([st["sen"], st["sea"]])

    all_means = np.asarray(all_means, dtype=float)
    all_stds = np.asarray(all_stds, dtype=float)
    finite = np.isfinite(all_means) & np.isfinite(all_stds)
    if np.any(finite):
        y_min = float(np.min(all_means[finite] - all_stds[finite]))
        y_max = float(np.max(all_means[finite] + all_stds[finite]))
    else:
        y_min, y_max = -0.1, 1.0
    # Keep a small negative margin to show bars below zero without creating huge empty space
    y_min = min(-0.1, y_min - 0.05)
    y_max = max(0.2, y_max + 0.10)
    # Add headroom so that the (inside) upper-right legend does not occlude error bars.
    if np.isfinite(y_max) and y_max > 0:
        y_max = y_max * 1.18
    else:
        y_max = y_max + 0.15

    for ax, (_, sp) in zip([ax1, ax2], specs.items()):
        # bars: for each method, two bars (EnKF, EAKF)
        labels = []
        means = []
        stds = []
        colors = []

        for disp, key, c_en, c_ea in methods:
            path = sp[key]
            if not path.exists():
                print(f"[plot_delta_rmse_comparison] missing: {path}")
                continue
            df = _read_seed_rows(path)
            if df.empty:
                continue
            st = _delta_stats(df)

            labels.extend([f"EnKF+{disp}", f"EAKF+{disp}"])
            means.extend([st["men"], st["mea"]])
            stds.extend([st["sen"], st["sea"]])
            colors.extend([c_en, c_ea])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0.0, color="gray", linewidth=1.0, alpha=0.7)
        ax.set_ylim((y_min, y_max))
        ax.set_title(sp["title"], fontsize=cfg.FONT_SIZE["title"], fontweight="bold")
        ax.set_ylabel("ΔRMSE (DA − DA+Model)", fontsize=cfg.FONT_SIZE["label"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=cfg.FONT_SIZE["tick"])
        ax.grid(True, alpha=cfg.GRID_ALPHA, axis="y")
        ax.tick_params(axis="both", which="major", labelsize=cfg.FONT_SIZE["tick"])

        for i, (m, s) in enumerate(zip(means, stds)):
            if np.isfinite(m):
                text = _fmt_val(m)
                ax.text(
                    i,
                    m + (s if np.isfinite(s) else 0.0) + 0.02,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=cfg.FONT_SIZE["annotation"],
                    fontweight="bold",
                )

                # Ensure near-zero bars are still visible: draw a small marker at the mean.
                if abs(float(m)) < 1e-3:
                    ax.plot([i], [m], marker="_", markersize=14, color="black", linewidth=2)

        legend_elements = [
            mpatches.Patch(color="#90EE90", label="EnKF+TF"),
            mpatches.Patch(color="#87CEEB", label="EAKF+TF"),
            mpatches.Patch(color="#DDA0DD", label="EnKF+LSTM"),
            mpatches.Patch(color="#F0E68C", label="EAKF+LSTM"),
            mpatches.Patch(color="#FFB6C1", label="EnKF+BiLSTM"),
            mpatches.Patch(color="#FFA07A", label="EAKF+BiLSTM"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            ncol=2,
            fontsize=cfg.FONT_SIZE["legend"],
            framealpha=0.7,
        )

    plt.tight_layout(pad=1.5)
    cfg.save_fig(fig, a.out_name)
    plt.show()


if __name__ == "__main__":
    plot_delta_rmse_comparison()
