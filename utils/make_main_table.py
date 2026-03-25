import csv
from pathlib import Path

import pandas as pd


def _read_mean_std(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if "seed" not in df.columns:
        raise ValueError(f"CSV missing seed column: {csv_path}")

    df2 = df.copy()
    df2["seed"] = df2["seed"].astype(str)

    mean_row = df2[df2["seed"].str.lower() == "mean"]
    std_row = df2[df2["seed"].str.lower() == "std"]
    if mean_row.empty or std_row.empty:
        raise ValueError(f"CSV missing mean/std rows: {csv_path}")

    mean_row = mean_row.iloc[0]
    std_row = std_row.iloc[0]

    cols = [
        ("EnKF", "rmse_enkf"),
        ("EAKF", "rmse_eakf"),
        ("Model", "rmse_tf"),
        ("Fused(EnKF)", "rmse_fused_enkf"),
        ("Fused(EAKF)", "rmse_fused_eakf"),
    ]

    out = {}
    for label, c in cols:
        if c not in df2.columns:
            raise ValueError(f"CSV missing column {c}: {csv_path}")
        m = float(mean_row[c])
        s = float(std_row[c])
        out[label] = (m, s)

    return out


def _fmt(m: float, s: float, digits: int = 3) -> str:
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main() -> None:
    root = Path(__file__).parent.resolve()
    result_dir = root / "result"
    paper_dir = root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        ("baseline", "Transformer"): result_dir / "multiseed_summary_w5_aEn0p33_aEa0p44.csv",
        ("baseline", "LSTM"): result_dir / "multiseed_summary_lstm_w5_aEn0p33_aEa0p44.csv",
        ("sparse", "Transformer"): result_dir / "multiseed_summary_w5_aEn0p2_aEa0p25.csv",
        ("sparse", "LSTM"): result_dir / "multiseed_summary_lstm_w5_aEn0p2_aEa0p25.csv",
    }

    rows = []
    for (scenario, model), p in paths.items():
        stats = _read_mean_std(p)
        row = {
            "scenario": scenario,
            "model": model,
        }
        for k, (m, s) in stats.items():
            row[k] = _fmt(m, s, digits=3)
        rows.append(row)

    out_cols = ["scenario", "model", "EnKF", "EAKF", "Model", "Fused(EnKF)", "Fused(EAKF)"]

    # CSV
    out_csv = paper_dir / "main_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in out_cols})

    # LaTeX
    out_tex = paper_dir / "main_table.tex"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results (mean$\pm$std over seeds).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")
    lines.append(r"Scenario & Model & EnKF & EAKF & Model & Fused(EnKF) & Fused(EAKF) \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['scenario']} & {r['model']} & {r['EnKF']} & {r['EAKF']} & {r['Model']} & {r['Fused(EnKF)']} & {r['Fused(EAKF)']} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
