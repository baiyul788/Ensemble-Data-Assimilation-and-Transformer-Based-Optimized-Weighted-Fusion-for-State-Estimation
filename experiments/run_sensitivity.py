"""
参数敏感性分析脚本

控制变量法：每次只变一个参数，其他固定在基准值

基准配置 (sparse):
    obs_dim=6, N=10, sigma_obs=2.0, sigma_model=0.1

测试参数:
    obs_dim:     6, 9, 18
    N:           10, 20, 40
    sigma_obs:   1.0, 2.0, 4.0
    sigma_model: 0.0, 0.1, 0.2

运行方式:
    py run_sensitivity.py              # 运行全部敏感性测试
    py run_sensitivity.py --param s    # 只测试 obs_dim
    py run_sensitivity.py --param N    # 只测试集合规模
    py run_sensitivity.py --summary    # 只生成汇总表（不重跑实验）
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
import shutil

from typing import Optional

import numpy as np

# ============================================================
# Baseline configuration
# ============================================================
BASELINE = {
    "obs_dim": 6,
    "N": 10,
    "sigma_obs": 2.0,
    "sigma_model": 0.1,
}

# Fixed parameters
TOTAL_TIME = 20.0
DT_OBS = 0.2
WINDOW = 5
EPOCHS = 300
ALPHA_ENKF = 0.20
ALPHA_EAKF = 0.25
N_SEEDS = 5
SEEDS = list(range(N_SEEDS))

# Sensitivity parameters
SENSITIVITY_PARAMS = {
    "s": {"name": "obs_dim", "values": [6, 9, 18], "label": "观测维度"},
    "N": {"name": "N", "values": [10, 20, 40], "label": "集合规模"},
    "sigma_obs": {"name": "sigma_obs", "values": [1.0, 2.0, 4.0], "label": "观测噪声"},
    "sigma_model": {"name": "sigma_model", "values": [0.0, 0.1, 0.2], "label": "模型误差"},
}

DATA_DIR = Path("data")
RESULT_DIR = Path("result")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def _make_tag(param_name: str, param_value) -> str:
    """生成实验标签"""
    return f"sens_{param_name}{_slug_float(float(param_value)) if isinstance(param_value, float) else param_value}"


def run_cmd(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'='*60}")
    if desc:
        print(f"[{desc}]")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def _base_csv_path() -> Path:
    return RESULT_DIR / f"multiseed_summary_w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}.csv"


def _tagged_csv_path(tag: str) -> Path:
    return RESULT_DIR / f"sensitivity_{tag}_w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}.csv"


def run_single_experiment(
    obs_dim: int,
    N: int,
    sigma_obs: float,
    sigma_model: float,
    tag: str,
) -> None:
    """运行单组实验（generate + train + evaluate）"""
    
    seeds_str = ",".join(map(str, SEEDS))
    
    # Step 1: Generate
    print(f"\n--- 生成数据集: {tag} ---")
    cmd = [
        sys.executable,
        "testing/test.py",
        "--seeds", seeds_str,
        "--total_time", str(TOTAL_TIME),
        "--dt_obs", str(DT_OBS),
        "--N", str(N),
        "--sigma_obs", str(sigma_obs),
        "--sigma_model", str(sigma_model),
        "--obs_dim", str(obs_dim),
        "--skip_eval",
        "--tag", tag,
    ]
    ret = run_cmd(cmd, f"生成数据集 {tag}")
    if ret != 0:
        print(f"警告: {tag} 数据集生成失败")
        return None
    
    # Step 2: Train
    print(f"\n--- 训练模型: {tag} ---")
    glob_pattern = (
        f"date/dataset_l96_n36_dt{_slug_float(0.01)}_T{_slug_float(TOTAL_TIME)}_"
        f"dtobs{_slug_float(DT_OBS)}_s{obs_dim}_N{N}_"
        f"sigObs{_slug_float(sigma_obs)}_sigModel{_slug_float(sigma_model)}_seed*_{tag}.npz"
    )
    cmd = [
        sys.executable,
        "training/train.py",
        "--glob", glob_pattern,
        "--epochs", str(EPOCHS),
        "--window", str(WINDOW),
        "--device", "auto",
    ]
    ret = run_cmd(cmd, f"训练模型 {tag}")
    if ret != 0:
        print(f"警告: {tag} 训练失败")
        return None
    
    # Step 3: Evaluate
    print(f"\n--- 评估融合: {tag} ---")
    cmd = [
        sys.executable,
        "testing/test.py",
        "--seeds", seeds_str,
        "--total_time", str(TOTAL_TIME),
        "--dt_obs", str(DT_OBS),
        "--N", str(N),
        "--sigma_obs", str(sigma_obs),
        "--sigma_model", str(sigma_model),
        "--obs_dim", str(obs_dim),
        "--fusion_window", str(WINDOW),
        "--alpha_enkf", str(ALPHA_ENKF),
        "--alpha_eakf", str(ALPHA_EAKF),
        "--export_csv",
        "--tag", tag,
    ]
    ret = run_cmd(cmd, f"评估融合 {tag}")
    if ret != 0:
        print(f"警告: {tag} 评估失败")
        return
    
    # Read results
    src_csv = _base_csv_path()
    if not src_csv.exists():
        print(f"警告: 未找到 CSV {src_csv}")
        return

    dst_csv = _tagged_csv_path(tag)
    shutil.copy(str(src_csv), str(dst_csv))
    print(f"CSV 已复制: {dst_csv}")
    return


def parse_csv_results(tag: str) -> Optional[dict]:
    """从 CSV 读取结果"""
    csv_path = _tagged_csv_path(tag)
    if not csv_path.exists():
        return None
    
    results = {
        "rmse_enkf": [],
        "rmse_eakf": [],
        "rmse_fused_enkf": [],
        "rmse_fused_eakf": [],
    }
    
    with open(str(csv_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("seed") in ("mean", "std", "-"):
                continue
            try:
                seed = int(row["seed"])
                if seed not in SEEDS:
                    continue
                results["rmse_enkf"].append(float(row["rmse_enkf"]))
                results["rmse_eakf"].append(float(row["rmse_eakf"]))
                results["rmse_fused_enkf"].append(float(row["rmse_fused_enkf"]))
                results["rmse_fused_eakf"].append(float(row["rmse_fused_eakf"]))
            except (ValueError, KeyError):
                continue
    
    if len(results["rmse_enkf"]) == 0:
        return None
    
    def mean_std(arr):
        arr = np.array([x for x in arr if np.isfinite(x)])
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    
    m_enkf, s_enkf = mean_std(results["rmse_enkf"])
    m_eakf, s_eakf = mean_std(results["rmse_eakf"])
    m_fen, s_fen = mean_std(results["rmse_fused_enkf"])
    m_fea, s_fea = mean_std(results["rmse_fused_eakf"])
    
    improve_enkf = (m_enkf - m_fen) / m_enkf * 100 if m_enkf > 0 else float("nan")
    improve_eakf = (m_eakf - m_fea) / m_eakf * 100 if m_eakf > 0 else float("nan")
    
    return {
        "rmse_enkf": m_enkf,
        "rmse_eakf": m_eakf,
        "rmse_fused_enkf": m_fen,
        "rmse_fused_eakf": m_fea,
        "std_enkf": s_enkf,
        "std_eakf": s_eakf,
        "std_fused_enkf": s_fen,
        "std_fused_eakf": s_fea,
        "improve_enkf": improve_enkf,
        "improve_eakf": improve_eakf,
        "valid_seeds": int(len(results["rmse_enkf"])),
    }


def run_sensitivity_for_param(param_key: str) -> list[dict]:
    """对单个参数进行敏感性测试"""
    param_info = SENSITIVITY_PARAMS[param_key]
    param_name = param_info["name"]
    values = param_info["values"]
    
    print(f"\n{'='*70}")
    print(f"敏感性测试: {param_info['label']} ({param_name})")
    print(f"测试值: {values}")
    print("=" * 70)
    
    results = []
    
    for val in values:
        # Build config
        cfg = dict(BASELINE)
        cfg[param_name] = val
        tag = _make_tag(param_key, val)
        
        print(f"\n>>> 测试 {param_name}={val} (tag={tag})")
        
        # Run experiment
        run_single_experiment(
            obs_dim=cfg["obs_dim"],
            N=cfg["N"],
            sigma_obs=cfg["sigma_obs"],
            sigma_model=cfg["sigma_model"],
            tag=tag,
        )
        
        # Parse results
        res = parse_csv_results(tag)
        if res:
            res["param"] = param_key
            res["value"] = val
            res["tag"] = tag
            results.append(res)
    
    return results


def print_summary_table(all_results: list[dict]):
    """打印汇总表格"""
    print("\n" + "=" * 100)
    print("参数敏感性分析汇总")
    print("=" * 100)
    print(f"{'参数':<12} {'值':<8} {'有效':<6} {'EnKF':<12} {'Fused(EnKF)':<12} {'提升%':<8} {'EAKF':<12} {'Fused(EAKF)':<12} {'提升%':<8}")
    print("-" * 100)
    
    for r in all_results:
        print(
            f"{r['param']:<12} "
            f"{r['value']:<8} "
            f"{int(r.get('valid_seeds', 0)):<6d} "
            f"{r['rmse_enkf']:.3f}±{r['std_enkf']:.2f}  "
            f"{r['rmse_fused_enkf']:.3f}±{r['std_fused_enkf']:.2f}  "
            f"{r['improve_enkf']:>6.1f}%  "
            f"{r['rmse_eakf']:.3f}±{r['std_eakf']:.2f}  "
            f"{r['rmse_fused_eakf']:.3f}±{r['std_fused_eakf']:.2f}  "
            f"{r['improve_eakf']:>6.1f}%"
        )
    
    print("=" * 100)
    
    # Save to CSV
    csv_path = RESULT_DIR / "sensitivity_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "param", "value", 
            "valid_seeds",
            "rmse_enkf", "std_enkf", "rmse_fused_enkf", "std_fused_enkf", "improve_enkf",
            "rmse_eakf", "std_eakf", "rmse_fused_eakf", "std_fused_eakf", "improve_eakf",
        ])
        for r in all_results:
            writer.writerow([
                r["param"], r["value"],
                int(r.get("valid_seeds", 0)),
                r["rmse_enkf"], r["std_enkf"], r["rmse_fused_enkf"], r["std_fused_enkf"], r["improve_enkf"],
                r["rmse_eakf"], r["std_eakf"], r["rmse_fused_eakf"], r["std_fused_eakf"], r["improve_eakf"],
            ])
    print(f"\n汇总表已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="参数敏感性分析")
    parser.add_argument(
        "--param",
        choices=list(SENSITIVITY_PARAMS.keys()) + ["all"],
        default="all",
        help="测试哪个参数 (s/N/sigma_obs/sigma_model/all)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="只生成汇总表，不重跑实验",
    )
    args = parser.parse_args()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if args.summary:
        # Only read existing results
        print("读取已有实验结果...")
        for param_key, param_info in SENSITIVITY_PARAMS.items():
            for val in param_info["values"]:
                tag = _make_tag(param_key, val)
                res = parse_csv_results(tag)
                if res:
                    res["param"] = param_key
                    res["value"] = val
                    res["tag"] = tag
                    all_results.append(res)
    else:
        # Run experiments
        if args.param == "all":
            for param_key in SENSITIVITY_PARAMS:
                results = run_sensitivity_for_param(param_key)
                all_results.extend(results)
        else:
            results = run_sensitivity_for_param(args.param)
            all_results.extend(results)
    
    if all_results:
        print_summary_table(all_results)
    else:
        print("没有找到有效的实验结果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
