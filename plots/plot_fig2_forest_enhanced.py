import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_config as cfg

def _auto_sparse_tf_csv() -> str:
    # Only sparse config (s6, N10, sigModel0p1). This avoids mixing baseline loc/infl CSVs.
    cands = sorted(
        list(cfg.RESULT_DIR.glob("plan_a_multiseed_s6_N10_*_sigModel0p1_*_loc*_infl*.csv")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    )
    if not cands:
        raise FileNotFoundError(
            "Missing sparse transformer CSV with localization/inflation. Expected something like: "
            "result/plan_a_multiseed_*_loc*_infl*.csv"
        )

    # Prefer the final recommended setting: infl=1.00 (slug 'infl1').
    preferred = [p for p in cands if "_infl1.csv" in p.name]
    if len(preferred) == 1:
        return preferred[0].name
    if len(preferred) > 1:
        names = [p.name for p in preferred]
        raise FileExistsError(
            f"Multiple sparse infl=1 candidates found: {names}. Please keep only one (e.g., loc6_infl1)."
        )

    names = [p.name for p in cands]
    raise FileExistsError(
        f"Multiple sparse loc/infl CSVs found (no unique infl=1): {names}. "
        f"Please keep only the final one (recommended: *_loc6_infl1.csv)."
    )

def load_all_data():
    sparse_tf_csv = _auto_sparse_tf_csv()
    files = {
        ('baseline', 'transformer'): cfg.RESULT_DIR / "multiseed_summary_w5_aEn0p33_aEa0p44.csv",
        ('baseline', 'lstm'): cfg.RESULT_DIR / "multiseed_summary_lstm_w5_aEn0p33_aEa0p44.csv", 
        ('baseline', 'bilstm'): cfg.RESULT_DIR / "multiseed_summary_bilstm_w5_aEn0p33_aEa0p44.csv",
        ('sparse', 'transformer'): cfg.RESULT_DIR / sparse_tf_csv,
        ('sparse', 'lstm'): cfg.RESULT_DIR / "multiseed_summary_lstm_w5_aEn0p2_aEa0p25.csv",
        ('sparse', 'bilstm'): cfg.RESULT_DIR / "multiseed_summary_bilstm_w5_aEn0p2_aEa0p25.csv",
    }
    
    data = {}
    for (config, model), filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required CSV: {filepath}")
        df = pd.read_csv(filepath)
        numeric_df = df[pd.to_numeric(df['seed'], errors='coerce').notna()].copy()
        data[(config, model)] = numeric_df
    
    return data

def calculate_stats(values):
    values = np.array(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0

def extract_all_stats(data, config):
    stats = {}
    
    if (config, 'transformer') in data:
        df = data[(config, 'transformer')]
        stats['EnKF'] = calculate_stats(df['rmse_enkf'].values)
        stats['EAKF'] = calculate_stats(df['rmse_eakf'].values)
    
    neural_models = [('transformer', 'TF'), ('lstm', 'LSTM'), ('bilstm', 'BiLSTM')]
    for model_key, model_name in neural_models:
        if (config, model_key) in data:
            df = data[(config, model_key)]
            stats[f'EnKF-{model_name}'] = calculate_stats(df['rmse_fused_enkf'].values)
            stats[f'EAKF-{model_name}'] = calculate_stats(df['rmse_fused_eakf'].values)
    
    return stats

data = load_all_data()

method_groups = [
    ('EnKF', cfg.get_color("enkf"), 'Traditional DA'),
    ('EAKF', cfg.get_color("eakf"), 'Traditional DA'),
    
    ('EnKF-TF', '#90EE90', 'Fusion (TF)'),
    ('EAKF-TF', '#87CEEB', 'Fusion (TF)'),
    
    ('EnKF-LSTM', '#DDA0DD', 'Fusion (LSTM)'),
    ('EAKF-LSTM', '#F0E68C', 'Fusion (LSTM)'),
    
    ('EnKF-BiLSTM', '#FFB6C1', 'Fusion (BiLSTM)'),
    ('EAKF-BiLSTM', '#FFA07A', 'Fusion (BiLSTM)'),
]

methods = [item[0] for item in method_groups]
colors = {item[0]: item[1] for item in method_groups}

sparse_stats = extract_all_stats(data, 'sparse')
baseline_stats = extract_all_stats(data, 'baseline')

fig, axes = plt.subplots(1, 2, figsize=cfg.get_figsize("forest"), sharey=True)

configs = [
    ("Sparse Configuration", sparse_stats, axes[0]),
    ("Baseline Configuration", baseline_stats, axes[1]),
]

for title, stats, ax in configs:
    y_pos = np.arange(len(methods))
    
    for i, method in enumerate(methods):
        if method in stats:
            mean, std = stats[method]
            if not np.isnan(mean):
                color = colors[method]
                
                ax.errorbar(mean, i, xerr=std, fmt="o", color=color, 
                           markersize=8, capsize=4, capthick=1.5, elinewidth=1.5)
                
                ax.annotate(f"{mean:.3f}±{std:.3f}", xy=(mean + std + 0.05, i),
                           va="center", ha="left",
                           fontsize=cfg.FONT_SIZE["annotation"], color="black")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=cfg.FONT_SIZE["tick"])
    ax.set_xlabel("RMSE", fontsize=cfg.FONT_SIZE["label"])
    ax.set_title(title, fontsize=cfg.FONT_SIZE["title"], fontweight='bold')
    ax.grid(True, alpha=cfg.GRID_ALPHA, axis="x")
    
    ax.set_xlim(2.5, 5.5)

import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color=cfg.get_color("enkf"), label='EnKF'),
    mpatches.Patch(color=cfg.get_color("eakf"), label='EAKF'),
    mpatches.Patch(color='#90EE90', label='EnKF-TF'),
    mpatches.Patch(color='#87CEEB', label='EAKF-TF'),
    mpatches.Patch(color='#DDA0DD', label='EnKF-LSTM'),
    mpatches.Patch(color='#F0E68C', label='EAKF-LSTM'),
    mpatches.Patch(color='#FFB6C1', label='EnKF-BiLSTM'),
    mpatches.Patch(color='#FFA07A', label='EAKF-BiLSTM'),
]
axes[1].legend(handles=legend_elements, loc='upper right', ncol=2,
              fontsize=cfg.FONT_SIZE["legend"]-1, framealpha=0.9)

group_boundaries = [1.5, 3.5, 5.5]
for ax in axes:
    for boundary in group_boundaries:
        ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout(pad=1.5)
cfg.save_fig(fig, "fig2_forest_enhanced")
plt.show()