"""
路线 A：论文最小闭环实验
目标：多 seeds 下各方法的 mean±std，验证融合是否稳定提升

运行方式：
    py run_experiment.py --step all        # 运行全部步骤
    py run_experiment.py --step generate   # 只生成数据集
    py run_experiment.py --step train      # 只训练模型
    py run_experiment.py --step evaluate   # 只评估融合（生成正确的 fused RMSE）
    py run_experiment.py --step summary    # 只生成汇总表（从 evaluate 的 CSV 读取）

配置说明：
    --config baseline  : obs_dim=9, σ_obs=2.0, σ_model=0.0, N=20 (默认)
    --config sparse    : obs_dim=6, σ_obs=2.0, σ_model=0.1, N=10 (更有利于融合)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

# ============================================================
# Experiment configuration presets
# ============================================================
CONFIGS = {
    # Baseline config
    "baseline": {
        "obs_dim": 9,
        "sigma_obs": 2.0,
        "sigma_model": 0.0,
        "N": 20,
        "alpha_enkf": 0.33,
        "alpha_eakf": 0.44,
        "localization_radius": None,
        "inflation_factor": None,
    },
    # Sparse obs + model error
    "sparse": {
        "obs_dim": 6,
        "sigma_obs": 2.0,
        "sigma_model": 0.1,
        "N": 10,
        "alpha_enkf": 0.20,
        "alpha_eakf": 0.25,
        "localization_radius": None,
        "inflation_factor": None,
    },
}

# Default config
CONFIG_NAME = "baseline"

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))

# Fixed experiment parameters
TOTAL_TIME = 20.0
DT_OBS = 0.2

# Transformer training parameters
WINDOW = 5
EPOCHS = 300
HIDDEN = 128
ATTN_HEADS = 4
N_LAYERS = 3

# Loaded from config (updated in main() based on --config)
OBS_DIM = CONFIGS[CONFIG_NAME]["obs_dim"]
SIGMA_OBS = CONFIGS[CONFIG_NAME]["sigma_obs"]
SIGMA_MODEL = CONFIGS[CONFIG_NAME]["sigma_model"]
N_ENS = CONFIGS[CONFIG_NAME]["N"]
ALPHA_ENKF = CONFIGS[CONFIG_NAME]["alpha_enkf"]
ALPHA_EAKF = CONFIGS[CONFIG_NAME]["alpha_eakf"]
LOCALIZATION_RADIUS = CONFIGS[CONFIG_NAME]["localization_radius"]
INFLATION_FACTOR = CONFIGS[CONFIG_NAME]["inflation_factor"]

# Paths
DATA_DIR = Path("date")
RESULT_DIR = Path("result")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _print_config_banner():
    """打印当前实验配置"""
    print("\n" + "=" * 70)
    print("                    实验配置 (Plan A)")
    print("=" * 70)
    print(f"  配置名称:     {CONFIG_NAME}")
    print("-" * 70)
    print("  物理参数:")
    print(f"    状态维度 n:       36 (Lorenz-96)")
    print(f"    观测维度 s:       {OBS_DIM} ({OBS_DIM/36*100:.1f}% 观测率)")
    print(f"    观测噪声 σ_obs:   {SIGMA_OBS}")
    print(f"    模型误差 σ_model: {SIGMA_MODEL}")
    print(f"    集合规模 N:       {N_ENS}")
    print(f"    观测间隔 dt_obs:  {DT_OBS}")
    print(f"    总时长 T:         {TOTAL_TIME}")
    print("-" * 70)
    print("  Transformer 参数:")
    print(f"    窗口大小 window:  {WINDOW}")
    print(f"    隐藏层维度:       {HIDDEN}")
    print(f"    注意力头数:       {ATTN_HEADS}")
    print(f"    层数:             {N_LAYERS}")
    print("-" * 70)
    print("  融合参数:")
    print(f"    α_enkf:           {ALPHA_ENKF}")
    print(f"    α_eakf:           {ALPHA_EAKF}")
    print("-" * 70)
    print("  DA 调参:")
    print(f"    localization_radius: {LOCALIZATION_RADIUS}")
    print(f"    inflation_factor:    {INFLATION_FACTOR}")
    print("-" * 70)
    print(f"  随机种子数:         {N_SEEDS} (seeds: {SEEDS})")
    print("=" * 70 + "\n")


def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def run_cmd(cmd: list[str], desc: str = "") -> int:
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    if desc:
        print(f"[{desc}]")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def step_generate():
    """
    步骤 1：生成多 seed 的数据集
    使用 --skip_eval 只生成数据，不评估融合
    """
    print("\n" + "=" * 60)
    print(f"步骤 1：生成数据集 ({N_SEEDS} seeds)")
    print(f"  配置: s={OBS_DIM}, σ_obs={SIGMA_OBS}, σ_model={SIGMA_MODEL}, N={N_ENS}")
    print("=" * 60)

    seeds_str = ",".join(map(str, SEEDS))
    cmd = [
        sys.executable,
        "testing/test.py",
        "--seeds", seeds_str,
        "--total_time", str(TOTAL_TIME),
        "--dt_obs", str(DT_OBS),
        "--N", str(N_ENS),
        "--sigma_obs", str(SIGMA_OBS),
        "--sigma_model", str(SIGMA_MODEL),
        "--skip_eval",
        "--tag", CONFIG_NAME,
    ]
    if LOCALIZATION_RADIUS is not None:
        cmd.extend(["--localization_radius", str(LOCALIZATION_RADIUS)])
    if INFLATION_FACTOR is not None:
        cmd.extend(["--inflation_factor", str(INFLATION_FACTOR)])
    if OBS_DIM > 0:
        cmd.extend(["--obs_dim", str(OBS_DIM)])

    ret = run_cmd(cmd, "生成数据集")
    if ret != 0:
        print("警告: 数据集生成失败")

    return ret


def step_train():
    """
    步骤 2：训练 Transformer 模型
    只训练与当前配置匹配的数据集
    """
    print("\n" + "=" * 60)
    print(f"步骤 2：训练 Transformer 模型 ({N_SEEDS} seeds)")
    print(f"  配置: s={OBS_DIM}, σ_obs={SIGMA_OBS}, σ_model={SIGMA_MODEL}, N={N_ENS}")
    print("=" * 60)

    # Build an exact-match glob pattern
    glob_pattern = (
        f"date/dataset_l96_n36_dt{_slug_float(0.01)}_T{_slug_float(TOTAL_TIME)}_"
        f"dtobs{_slug_float(DT_OBS)}_s{OBS_DIM}_N{N_ENS}_"
        f"sigObs{_slug_float(SIGMA_OBS)}_sigModel{_slug_float(SIGMA_MODEL)}_seed*_{CONFIG_NAME}.npz"
    )

    cmd = [
        sys.executable,
        "training/train.py",
        "--glob", glob_pattern,
        "--epochs", str(EPOCHS),
        "--window", str(WINDOW),
        "--hidden", str(HIDDEN),
        "--attn_heads", str(ATTN_HEADS),
        "--n_layers", str(N_LAYERS),
        "--device", "auto",
    ]

    return run_cmd(cmd, "批量训练模型")


def step_evaluate():
    """
    步骤 3：评估融合效果
    调用 test.py --export_csv，生成包含正确 fused RMSE 的 CSV
    然后重命名 CSV 文件，加上完整实验配置参数，避免被覆盖
    """
    print("\n" + "=" * 60)
    print("步骤 3：评估融合效果")
    print(f"  配置: s={OBS_DIM}, σ_obs={SIGMA_OBS}, σ_model={SIGMA_MODEL}, N={N_ENS}")
    print(f"  融合权重: α_enkf={ALPHA_ENKF}, α_eakf={ALPHA_EAKF}")
    print("=" * 60)

    seeds_str = ",".join(map(str, SEEDS))
    cmd = [
        sys.executable,
        "testing/test.py",
        "--seeds", seeds_str,
        "--total_time", str(TOTAL_TIME),
        "--dt_obs", str(DT_OBS),
        "--N", str(N_ENS),
        "--sigma_obs", str(SIGMA_OBS),
        "--sigma_model", str(SIGMA_MODEL),
        "--fusion_window", str(WINDOW),
        "--alpha_enkf", str(ALPHA_ENKF),
        "--alpha_eakf", str(ALPHA_EAKF),
        "--export_csv",
        "--tag", CONFIG_NAME,
    ]
    if LOCALIZATION_RADIUS is not None:
        cmd.extend(["--localization_radius", str(LOCALIZATION_RADIUS)])
    if INFLATION_FACTOR is not None:
        cmd.extend(["--inflation_factor", str(INFLATION_FACTOR)])
    if OBS_DIM > 0:
        cmd.extend(["--obs_dim", str(OBS_DIM)])

    ret = run_cmd(cmd, "评估融合效果")

    # Copy CSV with full config in name
    src_csv = RESULT_DIR / f"multiseed_summary_w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}.csv"
    loc_part = f"_loc{_slug_float(LOCALIZATION_RADIUS)}" if LOCALIZATION_RADIUS is not None else ""
    infl_part = f"_infl{_slug_float(INFLATION_FACTOR)}" if INFLATION_FACTOR is not None else ""
    dst_csv = RESULT_DIR / (
        f"plan_a_multiseed_"
        f"s{OBS_DIM}_N{N_ENS}_"
        f"sigObs{_slug_float(SIGMA_OBS)}_sigModel{_slug_float(SIGMA_MODEL)}_"
        f"T{_slug_float(TOTAL_TIME)}_dtobs{_slug_float(DT_OBS)}_"
        f"w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}_"
        f"seeds{N_SEEDS}{loc_part}{infl_part}.csv"
    )

    if src_csv.exists():
        import shutil
        shutil.copy(str(src_csv), str(dst_csv))
        print(f"\nCSV 已复制: {dst_csv}")

    return ret


def step_summary():
    """
    步骤 4：生成汇总表格
    优先读取带完整配置的 CSV，否则回退到原始 CSV
    """
    print("\n" + "=" * 60)
    print("步骤 4：生成汇总表格")
    print("=" * 60)

    # Prefer full-config CSV (generated by step_evaluate)
    csv_path = RESULT_DIR / (
        f"plan_a_multiseed_"
        f"s{OBS_DIM}_N{N_ENS}_"
        f"sigObs{_slug_float(SIGMA_OBS)}_sigModel{_slug_float(SIGMA_MODEL)}_"
        f"T{_slug_float(TOTAL_TIME)}_dtobs{_slug_float(DT_OBS)}_"
        f"w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}_"
        f"seeds{N_SEEDS}.csv"
    )

    # Fallback to original CSV
    if not csv_path.exists():
        csv_path = RESULT_DIR / f"multiseed_summary_w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}.csv"

    if not csv_path.exists():
        print(f"未找到 CSV 文件")
        print("请先运行 evaluate 步骤: py run_experiment.py --step evaluate")
        return 1

    print(f"读取 CSV: {csv_path}")

    # Read CSV
    results = {
        "seed": [],
        "rmse_tf": [],
        "rmse_enkf": [],
        "rmse_eakf": [],
        "rmse_fused_enkf": [],
        "rmse_fused_eakf": [],
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip mean/std rows
            if row.get("seed") in ("mean", "std", "-"):
                continue
            try:
                seed = int(row["seed"])
                # Only collect selected seeds
                if seed not in SEEDS:
                    continue

                results["seed"].append(seed)
                results["rmse_tf"].append(float(row["rmse_tf"]))
                results["rmse_enkf"].append(float(row["rmse_enkf"]))
                results["rmse_eakf"].append(float(row["rmse_eakf"]))
                results["rmse_fused_enkf"].append(float(row["rmse_fused_enkf"]))
                results["rmse_fused_eakf"].append(float(row["rmse_fused_eakf"]))
            except (ValueError, KeyError) as e:
                print(f"跳过无效行: {row}, 错误: {e}")

    if len(results["seed"]) == 0:
        print("CSV 中没有有效的结果数据")
        return 1

    # Check if any seeds are missing
    missing_seeds = set(SEEDS) - set(results["seed"])
    if missing_seeds:
        print(f"警告: 以下 seeds 缺少结果: {sorted(missing_seeds)}")

    # Compute statistics
    def mean_std(arr):
        arr = np.array([x for x in arr if np.isfinite(x)])
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    print("\n" + "=" * 60)
    print("实验结果汇总 (Plan A)")
    print("=" * 60)
    print(f"实验配置:")
    print(f"  obs_dim={OBS_DIM}, σ_obs={SIGMA_OBS}, σ_model={SIGMA_MODEL}, N={N_ENS}")
    print(f"  window={WINDOW}, α_enkf={ALPHA_ENKF}, α_eakf={ALPHA_EAKF}")
    print(f"有效 seeds: {len(results['seed'])} / {N_SEEDS}")
    print("-" * 60)
    print(f"{'Method':<20} {'RMSE (mean ± std)':<25}")
    print("-" * 60)

    # Only show DA and fused methods
    methods = [
        ("EnKF", "rmse_enkf"),
        ("EAKF", "rmse_eakf"),
        ("Fused(EnKF)", "rmse_fused_enkf"),
        ("Fused(EAKF)", "rmse_fused_eakf"),
    ]

    summary_data = {}
    for name, key in methods:
        m, s = mean_std(results[key])
        summary_data[name] = (m, s)
        print(f"{name:<20} {m:.6f} ± {s:.6f}")

    print("-" * 60)

    # Improvement analysis (DA baselines only)
    print("\n提升分析 (负值表示 RMSE 降低，即性能提升):")

    enkf_mean = summary_data["EnKF"][0]
    eakf_mean = summary_data["EAKF"][0]
    fused_enkf_mean = summary_data["Fused(EnKF)"][0]
    fused_eakf_mean = summary_data["Fused(EAKF)"][0]

    improve_enkf = None
    improve_eakf = None

    if np.isfinite(enkf_mean) and enkf_mean > 0 and np.isfinite(fused_enkf_mean):
        improve_enkf = (enkf_mean - fused_enkf_mean) / enkf_mean * 100
        print(f"  Fused(EnKF) vs EnKF:  RMSE 降低 {improve_enkf:.2f}%")

    if np.isfinite(eakf_mean) and eakf_mean > 0 and np.isfinite(fused_eakf_mean):
        improve_eakf = (eakf_mean - fused_eakf_mean) / eakf_mean * 100
        print(f"  Fused(EAKF) vs EAKF:  RMSE 降低 {improve_eakf:.2f}%")

    # Conclusion (DA baselines only)
    print("\n" + "-" * 60)
    print("结论:")

    if improve_enkf is not None and improve_eakf is not None:
        if improve_enkf > 0 and improve_eakf > 0:
            avg_improve = (improve_enkf + improve_eakf) / 2
            print(f"  ✅ 融合方法有效！")
            print(f"     Fused(EnKF) 比 EnKF 降低 {improve_enkf:.2f}%")
            print(f"     Fused(EAKF) 比 EAKF 降低 {improve_eakf:.2f}%")
            print(f"     平均提升: {avg_improve:.2f}%")
        elif improve_enkf > 0 or improve_eakf > 0:
            print(f"  ⚠️ 融合方法部分有效")
        else:
            print(f"  ❌ 融合方法未能提升 DA 性能")
    else:
        print(f"  ❓ 数据不完整，无法得出结论")

    # Save results to npz
    summary_path = RESULT_DIR / (
        f"plan_a_summary_"
        f"{CONFIG_NAME}_"
        f"s{OBS_DIM}_N{N_ENS}_"
        f"sigObs{_slug_float(SIGMA_OBS)}_sigModel{_slug_float(SIGMA_MODEL)}_"
        f"T{_slug_float(TOTAL_TIME)}_dtobs{_slug_float(DT_OBS)}_"
        f"w{WINDOW}_aEn{_slug_float(ALPHA_ENKF)}_aEa{_slug_float(ALPHA_EAKF)}_"
        f"seeds{N_SEEDS}.npz"
    )
    np.savez_compressed(
        str(summary_path),
        seeds=np.array(results["seed"], dtype=int),
        rmse_tf=np.array(results["rmse_tf"], dtype=float),
        rmse_enkf=np.array(results["rmse_enkf"], dtype=float),
        rmse_eakf=np.array(results["rmse_eakf"], dtype=float),
        rmse_fused_enkf=np.array(results["rmse_fused_enkf"], dtype=float),
        rmse_fused_eakf=np.array(results["rmse_fused_eakf"], dtype=float),
        summary_mean=np.array([summary_data[m][0] for m, _ in methods], dtype=float),
        summary_std=np.array([summary_data[m][1] for m, _ in methods], dtype=float),
        method_names=np.array([m for m, _ in methods], dtype=object),
        config=np.array({
            "config_name": str(CONFIG_NAME),
            "obs_dim": OBS_DIM,
            "sigma_obs": SIGMA_OBS,
            "sigma_model": SIGMA_MODEL,
            "N": N_ENS,
            "window": WINDOW,
            "alpha_enkf": ALPHA_ENKF,
            "alpha_eakf": ALPHA_EAKF,
            "localization_radius": LOCALIZATION_RADIUS,
            "inflation_factor": INFLATION_FACTOR,
            "total_time": TOTAL_TIME,
            "dt_obs": DT_OBS,
        }),
    )
    print(f"\n结果已保存: {summary_path}")

    return 0


def main():
    global CONFIG_NAME, OBS_DIM, SIGMA_OBS, SIGMA_MODEL, N_ENS, ALPHA_ENKF, ALPHA_EAKF, LOCALIZATION_RADIUS, INFLATION_FACTOR

    parser = argparse.ArgumentParser(description="Plan A: 论文最小闭环实验")
    parser.add_argument(
        "--step",
        choices=["all", "generate", "train", "evaluate", "summary"],
        default="all",
        help="运行哪个步骤",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="baseline",
        help="实验配置: baseline (默认) 或 sparse (更有利于融合)",
    )
    parser.add_argument("--localization_radius", type=float, default=None, help="DA localization radius (Gaspari-Cohn), None disables")
    parser.add_argument("--inflation_factor", type=float, default=None, help="DA multiplicative inflation factor, None disables")
    args = parser.parse_args()

    # Update global config
    CONFIG_NAME = args.config
    cfg = CONFIGS[CONFIG_NAME]
    OBS_DIM = cfg["obs_dim"]
    SIGMA_OBS = cfg["sigma_obs"]
    SIGMA_MODEL = cfg["sigma_model"]
    N_ENS = cfg["N"]
    ALPHA_ENKF = cfg["alpha_enkf"]
    ALPHA_EAKF = cfg["alpha_eakf"]
    LOCALIZATION_RADIUS = args.localization_radius if args.localization_radius is not None else cfg.get("localization_radius")
    INFLATION_FACTOR = args.inflation_factor if args.inflation_factor is not None else cfg.get("inflation_factor")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Print config
    _print_config_banner()

    if args.step == "all":
        ret = step_generate()
        if ret != 0:
            print("generate 步骤失败")
            return ret

        ret = step_train()
        if ret != 0:
            print("train 步骤失败")
            return ret

        ret = step_evaluate()
        if ret != 0:
            print("evaluate 步骤失败")
            return ret

        ret = step_summary()
        return ret

    elif args.step == "generate":
        return step_generate()
    elif args.step == "train":
        return step_train()
    elif args.step == "evaluate":
        return step_evaluate()
    elif args.step == "summary":
        return step_summary()


if __name__ == "__main__":
    sys.exit(main())
