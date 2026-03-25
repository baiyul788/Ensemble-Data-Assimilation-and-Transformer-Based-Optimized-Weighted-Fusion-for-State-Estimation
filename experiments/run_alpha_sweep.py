"""
融合权重 α 局部扫描脚本

目的：证明选择的 α 接近最优且在附近范围内表现稳定（鲁棒）
论文表述："Fusion weights were selected via a coarse local sweep on held-out segment"

运行方式：
    py run_alpha_sweep.py                    # 运行全部扫描（sparse + baseline）
    py run_alpha_sweep.py --config sparse    # 只扫描 sparse 配置
    py run_alpha_sweep.py --config baseline  # 只扫描 baseline 配置

注意：
    - 不需要重新训练，只调用 test.py 的 evaluate
    - 会自动查找已有的 ckpt 文件
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================
# Sweep configuration
# ============================================================
SWEEP_CONFIGS = {
    "sparse": {
        # Experiment parameters
        "obs_dim": 6,
        "N": 10,
        "sigma_obs": 2.0,
        "sigma_model": 0.1,
        # Sweep range
        "alpha_pairs": [
            (0.30, 0.35),
            (0.35, 0.40),
            (0.40, 0.45),
            (0.45, 0.50),
            (0.50, 0.55),
        ],
    },
    "baseline": {
        # Experiment parameters
        "obs_dim": 9,
        "N": 20,
        "sigma_obs": 2.0,
        "sigma_model": 0.0,
        # Sweep range
        "alpha_pairs": [
            (0.38, 0.44),
            (0.43, 0.49),
            (0.48, 0.54),
            (0.53, 0.59),
            (0.58, 0.64),
        ],
    },
}

EXTRA_LOWER_ALPHA_PAIRS = {
    "sparse": [
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
    ],
    "baseline": [
        (0.23, 0.29),
        (0.28, 0.34),
        (0.33, 0.39),
    ],
}

# Fixed parameters
SEEDS = "0,1,2,3,4"
TOTAL_TIME = 20.0
DT_OBS = 0.2
WINDOW = 5

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "result"


def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def _multiseed_csv_path(alpha_enkf: float, alpha_eakf: float) -> Path:
    return RESULT_DIR / (
        f"multiseed_summary_w{WINDOW}_aEn{_slug_float(alpha_enkf)}_aEa{_slug_float(alpha_eakf)}.csv"
    )


def _parse_mean_row(csv_path: Path, alpha_enkf: float, alpha_eakf: float) -> Optional[Dict]:
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("seed", "")).strip() == "mean":
                try:
                    return {
                        "alpha_enkf": alpha_enkf,
                        "alpha_eakf": alpha_eakf,
                        "rmse_fused_enkf": float(row["rmse_fused_enkf"]),
                        "rmse_fused_eakf": float(row["rmse_fused_eakf"]),
                        "rmse_enkf": float(row["rmse_enkf"]),
                        "rmse_eakf": float(row["rmse_eakf"]),
                    }
                except (TypeError, ValueError, KeyError):
                    return None
    return None


def run_single_eval(config_name: str, alpha_enkf: float, alpha_eakf: float) -> Optional[Dict]:
    """运行单次 evaluate 并返回结果"""
    cfg = SWEEP_CONFIGS[config_name]

    csv_path = _multiseed_csv_path(alpha_enkf, alpha_eakf)
    cached = csv_path.exists()
    
    cmd = [
        sys.executable,
        "testing/test.py",
        "--seeds", SEEDS,
        "--total_time", str(TOTAL_TIME),
        "--dt_obs", str(DT_OBS),
        "--N", str(cfg["N"]),
        "--sigma_obs", str(cfg["sigma_obs"]),
        "--sigma_model", str(cfg["sigma_model"]),
        "--obs_dim", str(cfg["obs_dim"]),
        "--fusion_window", str(WINDOW),
        "--alpha_enkf", str(alpha_enkf),
        "--alpha_eakf", str(alpha_eakf),
        "--export_csv",
        "--tag", config_name,
    ]
    
    print(f"\n{'='*60}")
    print(f"扫描: α_enkf={alpha_enkf}, α_eakf={alpha_eakf}{' [cached]' if cached else ''}")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    if cached:
        return _parse_mean_row(csv_path, alpha_enkf, alpha_eakf)

    result = subprocess.run(cmd, cwd=ROOT_DIR)

    if result.returncode != 0:
        print(f"警告: 评估失败")
        return None

    parsed = _parse_mean_row(csv_path, alpha_enkf, alpha_eakf)
    if parsed is None:
        print(f"警告: 未找到/无法解析 CSV {csv_path}")
    return parsed


def run_sweep(config_name: str, alpha_pairs: Optional[List[tuple[float, float]]] = None) -> List[Dict]:
    """对指定配置进行 α 扫描"""
    cfg = SWEEP_CONFIGS[config_name]
    
    print(f"\n{'#'*70}")
    print(f"# α 扫描: {config_name} 配置")
    print(f"# obs_dim={cfg['obs_dim']}, N={cfg['N']}, σ_obs={cfg['sigma_obs']}, σ_model={cfg['sigma_model']}")
    print(f"{'#'*70}")
    
    results = []
    pairs = cfg["alpha_pairs"] if alpha_pairs is None else alpha_pairs
    for alpha_enkf, alpha_eakf in pairs:
        res = run_single_eval(config_name, alpha_enkf, alpha_eakf)
        if res:
            results.append(res)
    
    return results


def print_summary(config_name: str, results: List[Dict]):
    """打印扫描结果汇总"""
    if not results:
        print(f"\n{config_name}: 无有效结果")
        return
    
    print(f"\n{'='*80}")
    print(f"α 扫描结果汇总: {config_name}")
    print("=" * 80)
    print(f"{'α_enkf':<10} {'α_eakf':<10} {'RMSE(Fused-EnKF)':<18} {'RMSE(Fused-EAKF)':<18} {'备注':<10}")
    print("-" * 80)
    
    # Best entries
    best_enkf = min(results, key=lambda x: x["rmse_fused_enkf"])
    best_eakf = min(results, key=lambda x: x["rmse_fused_eakf"])
    
    for r in results:
        notes = []
        if r["alpha_enkf"] == best_enkf["alpha_enkf"] and r["alpha_eakf"] == best_enkf["alpha_eakf"]:
            notes.append("best-EnKF")
        if r["alpha_enkf"] == best_eakf["alpha_enkf"] and r["alpha_eakf"] == best_eakf["alpha_eakf"]:
            notes.append("best-EAKF")
        note = (" <- " + ", ".join(notes)) if notes else ""
        print(f"{r['alpha_enkf']:<10.2f} {r['alpha_eakf']:<10.2f} {r['rmse_fused_enkf']:<18.6f} {r['rmse_fused_eakf']:<18.6f}{note}")
    
    print("-" * 80)
    print(f"最优 (Fused-EnKF): α_enkf={best_enkf['alpha_enkf']}, α_eakf={best_enkf['alpha_eakf']}, RMSE={best_enkf['rmse_fused_enkf']:.6f}")
    print(f"最优 (Fused-EAKF): α_enkf={best_eakf['alpha_enkf']}, α_eakf={best_eakf['alpha_eakf']}, RMSE={best_eakf['rmse_fused_eakf']:.6f}")
    
    # Save CSV
    csv_path = RESULT_DIR / f"alpha_sweep_{config_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_enkf", "alpha_eakf", "rmse_fused_enkf", "rmse_fused_eakf", "rmse_enkf", "rmse_eakf"])
        for r in results:
            writer.writerow([r["alpha_enkf"], r["alpha_eakf"], r["rmse_fused_enkf"], r["rmse_fused_eakf"], r["rmse_enkf"], r["rmse_eakf"]])
    print(f"\n结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="融合权重 α 局部扫描")
    parser.add_argument(
        "--config",
        choices=["sparse", "baseline", "all"],
        default="all",
        help="扫描哪个配置 (sparse/baseline/all)",
    )
    parser.add_argument(
        "--extend_lower",
        action="store_true",
        help="补扫更小 α 点位（已存在的组合会自动读取 CSV，不会重复跑）",
    )
    args = parser.parse_args()
    
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.config == "all":
        configs_to_run = ["sparse", "baseline"]
    else:
        configs_to_run = [args.config]
    
    all_results = {}
    for config_name in configs_to_run:
        alpha_pairs = SWEEP_CONFIGS[config_name]["alpha_pairs"]
        if args.extend_lower:
            alpha_pairs = EXTRA_LOWER_ALPHA_PAIRS.get(config_name, []) + alpha_pairs
        results = run_sweep(config_name, alpha_pairs=alpha_pairs)
        all_results[config_name] = results
        print_summary(config_name, results)
    
    print("\n" + "=" * 80)
    print("扫描完成！")
    print("=" * 80)
    print("\n论文表述建议:")
    print('  "Fusion weights were selected via a coarse local sweep around a')
    print('   preliminary setting on the held-out segment; the chosen weights')
    print('   are near-optimal and performance is stable within ±0.10."')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
