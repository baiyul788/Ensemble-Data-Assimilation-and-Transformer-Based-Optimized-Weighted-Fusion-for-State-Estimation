"""
轨迹数据准备脚本

功能：从 dataset + ckpt 生成包含 X_fused 的完整轨迹文件，供绘图使用

输入：
    - date/dataset_*.npz (含 X_true, X_enkf, X_eakf, Y)
    - result/transformer_ckpt_*.pt (Transformer 模型)
    - result/*multiseed*.csv (ckpt 路径索引)

输出：
    - result/trajectories_*.npz (含 X_true, X_enkf, X_eakf, X_tf, X_fused_enkf, X_fused_eakf, Y 等)

运行：
    py prepare_trajectories.py --config sparse
    py prepare_trajectories.py --config baseline
    py prepare_trajectories.py --csv result/sensitivity_sens_s6_w5_aEn0p4_aEa0p45.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Allow running from any working directory: add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.lstm import ObsToStateLSTM
from models.transformer import ObsToStateTransformer


# Repo root (for resolving relative paths)
ROOT_DIR = _REPO_ROOT

DATA_DIR = ROOT_DIR / "date"
RESULT_DIR = ROOT_DIR / "result"

# Fusion weights (aligned with run_experiment.py)
ALPHA = {
    "sparse": {"enkf": 0.20, "eakf": 0.25},
    "baseline": {"enkf": 0.33, "eakf": 0.44},
}


def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def load_model(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    """加载模型（Transformer/LSTM）"""
    # PyTorch compatibility: try/except fallback
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    model_type = str(ckpt.get("model_type", "transformer")).lower()
    
    if model_type == "lstm":
        model = ObsToStateLSTM(
            obs_dim=int(ckpt["obs_dim"]),
            state_dim=int(ckpt["state_dim"]),
            hidden=int(ckpt["hidden"]),
            n_layers=int(ckpt["n_layers"]),
            dropout=float(ckpt["dropout"]),
        )
    else:
        kwargs = dict(
            obs_dim=int(ckpt["obs_dim"]),
            state_dim=int(ckpt["state_dim"]),
            hidden=int(ckpt["hidden"]),
            attn_heads=int(ckpt.get("attn_heads", 4)),
            n_layers=int(ckpt["n_layers"]),
            dropout=float(ckpt["dropout"]),
        )
        # Compat with different model signatures: only pass max_len if supported
        sig = inspect.signature(ObsToStateTransformer.__init__)
        if "max_len" in sig.parameters:
            kwargs["max_len"] = max(512, int(ckpt["window"]))
 
        model = ObsToStateTransformer(**kwargs)
    
    # Legacy ckpt compat: LayerNorm key mapping (Transformer only)
    state = ckpt["model_state_dict"]
    if isinstance(state, dict) and model_type != "lstm":
        has_old_ln = any(k.endswith(".norm.a_2") or k.endswith(".norm.b_2") for k in state.keys())
        if has_old_ln:
            state2 = dict(state)
            for k, v in list(state.items()):
                if k.endswith(".norm.a_2"):
                    state2[k[:-4] + ".weight"] = v
                    state2.pop(k, None)
                elif k.endswith(".norm.b_2"):
                    state2[k[:-4] + ".bias"] = v
                    state2.pop(k, None)
            state = state2
    
    model.load_state_dict(state)
    model.eval()
    
    return model, ckpt


def make_windows(Y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """构造滑动窗口"""
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    T = Y.shape[0]
    if int(T) < int(window):
        raise ValueError("not enough samples for the given window")
    xs, idx = [], []
    for k in range(window - 1, T):
        xs.append(Y[k - window + 1 : k + 1, :])
        idx.append(k)
    return np.stack(xs, axis=0), np.array(idx, dtype=int)


def predict_model(model: torch.nn.Module, ckpt: dict, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用模型推理"""
    window = int(ckpt["window"])
    obs_dim = int(ckpt["obs_dim"])
    x_mean = np.asarray(ckpt["x_mean"], dtype=float)
    x_std = np.asarray(ckpt["x_std"], dtype=float)
    y_mean = np.asarray(ckpt["y_mean"], dtype=float)
    y_std = np.asarray(ckpt["y_std"], dtype=float)

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D [T, obs_dim], got shape={Y.shape}")
    if int(Y.shape[1]) != obs_dim:
        raise ValueError(f"obs_dim mismatch: Y has {int(Y.shape[1])} but ckpt expects {obs_dim}")
    
    Xw, idx = make_windows(Y, window)
    Xw_n = (Xw - x_mean) / x_std
    
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xw_n).float(), mask=None).cpu().numpy()
    
    pred = pred_n * y_std + y_mean
    return pred, idx


def _maybe_int_array_or_none(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object and x.item() is None:
        return None
    return np.asarray(x, dtype=int)


def process_single(dataset_path: Path, ckpt_path: Path, alpha_enkf: float, alpha_eakf: float) -> dict:
    """处理单个 dataset + ckpt"""
    # Load data
    with np.load(str(dataset_path), allow_pickle=True) as data:
        Y = np.asarray(data["Y"], dtype=float)
        X_true = np.asarray(data["X_true"], dtype=float)
        X_enkf = np.asarray(data["X_enkf"], dtype=float)
        X_eakf = np.asarray(data["X_eakf"], dtype=float)

        dt_obs = float(data["dt_obs"]) if "dt_obs" in data.files else float("nan")
        obs_indices = _maybe_int_array_or_none(data["obs_indices"]) if "obs_indices" in data.files else None
        ind_m = _maybe_int_array_or_none(data["ind_m"]) if "ind_m" in data.files else None
        seed = int(data["seed"]) if "seed" in data.files else -1
    
    # Load model and run inference
    model, ckpt = load_model(ckpt_path)
    model_type = str(ckpt.get("model_type", "transformer")).lower()
    X_tf, idx = predict_model(model, ckpt, Y)
    
    # Align indices
    X_true_aligned = X_true[idx]
    X_enkf_aligned = X_enkf[idx]
    X_eakf_aligned = X_eakf[idx]
    Y_aligned = Y[idx]
    
    # Compute fusion
    X_fused_enkf = alpha_enkf * X_enkf_aligned + (1 - alpha_enkf) * X_tf
    X_fused_eakf = alpha_eakf * X_eakf_aligned + (1 - alpha_eakf) * X_tf
    
    result = {
        "idx": idx,
        "Y": Y_aligned,
        "X_true": X_true_aligned,
        "X_enkf": X_enkf_aligned,
        "X_eakf": X_eakf_aligned,
        "X_tf": X_tf,
        "X_fused_enkf": X_fused_enkf,
        "X_fused_eakf": X_fused_eakf,
        "alpha_enkf": alpha_enkf,
        "alpha_eakf": alpha_eakf,
        "window": int(ckpt["window"]),
        "dt_obs": dt_obs,
        "seed": seed,
        "dataset": str(dataset_path),
        "ckpt": str(ckpt_path),
        "model_type": model_type,
    }
    
    if obs_indices is not None:
        result["obs_indices"] = obs_indices
    if ind_m is not None:
        result["ind_m"] = ind_m
    
    return result


def resolve_path(path_str: str) -> Path:
    """解析路径（处理相对路径）"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Try resolving from ROOT_DIR
    resolved = (ROOT_DIR / p).resolve()
    if resolved.exists():
        return resolved
    # Try original path
    if p.exists():
        return p.resolve()
    return resolved


def process_from_csv(csv_path: Path, alpha_enkf: float, alpha_eakf: float) -> None:
    """从 CSV 读取 ckpt 路径并处理"""
    result_dir = RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    
    with open(str(csv_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("seed") in ("mean", "std", "-"):
                continue
            
            seed_str = row.get("seed", "")
            if not seed_str:
                continue
            seed = int(seed_str)
            
            ckpt_str = row.get("ckpt", "")
            if not ckpt_str or ckpt_str == "-":
                continue
            
            ckpt_path = resolve_path(ckpt_str)
            if not ckpt_path.exists():
                print(f"跳过：ckpt 不存在 {ckpt_path}")
                continue

            # Prefer dataset column from CSV (written by test.py --export_csv)
            dataset_str = row.get("dataset", "")
            dataset_path: Optional[Path] = None
            dataset_stem: Optional[str] = None
            if dataset_str and dataset_str != "-":
                dataset_path = resolve_path(dataset_str)
                dataset_stem = dataset_path.stem
            else:
                # Fallback: infer dataset from ckpt name (legacy naming)
                ckpt_name = ckpt_path.stem
                if "_data" in ckpt_name and "_seed" in ckpt_name:
                    start = ckpt_name.index("_data") + 5
                    first_seed = ckpt_name.index("_seed", start)
                    # Find second _seed (training seed)
                    second_seed = ckpt_name.find("_seed", first_seed + 5)
                    if second_seed != -1:
                        end = second_seed
                    else:
                        # Fallback: find _w (window)
                        end = ckpt_name.find("_w", first_seed)
                        if end == -1:
                            end = first_seed
                    dataset_stem = ckpt_name[start:end]
                    dataset_path = DATA_DIR / f"{dataset_stem}.npz"
                else:
                    print(f"跳过：无法解析 dataset 路径 {ckpt_name}")
                    continue

            if not dataset_path.exists():
                print(f"跳过：dataset 不存在 {dataset_path}")
                continue

            print(f"处理 seed={seed}: {dataset_path.name}")
            
            result = process_single(dataset_path, ckpt_path, alpha_enkf, alpha_eakf)
            
            # Save
            if str(result.get("model_type", "transformer")).lower() == "transformer":
                out_name = f"trajectories_{dataset_stem}.npz"
            else:
                out_name = f"trajectories_{dataset_stem}_{str(result.get('model_type')).lower()}.npz"
            out_path = result_dir / out_name
            np.savez_compressed(str(out_path), **result)
            print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="准备轨迹数据供绘图")
    parser.add_argument("--config", choices=["sparse", "baseline"], help="使用预设配置")
    parser.add_argument("--csv", type=str, help="指定 CSV 文件路径")
    parser.add_argument("--alpha_enkf", type=float, default=0.4)
    parser.add_argument("--alpha_eakf", type=float, default=0.45)
    args = parser.parse_args()
    
    result_dir = RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    
    if args.config:
        alpha_enkf = ALPHA[args.config]["enkf"]
        alpha_eakf = ALPHA[args.config]["eakf"]
        
        # Find matching CSV
        if args.config == "sparse":
            csv_pattern = "plan_a_multiseed_s6_N10_*"
        else:
            csv_pattern = "plan_a_multiseed_s9_N20_*"
        
        csv_files = list(result_dir.glob(csv_pattern + ".csv"))
        if not csv_files:
            print(f"未找到 {args.config} 配置的 CSV")
            return 1

        slug_aen = _slug_float(alpha_enkf)
        slug_aea = _slug_float(alpha_eakf)
        matched = [
            p
            for p in csv_files
            if (f"_aEn{slug_aen}_aEa{slug_aea}_" in p.name)
        ]
        picked = matched if matched else csv_files
        picked = sorted(picked, key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = picked[0]
        print(f"使用配置: {args.config}, CSV: {csv_path}")
        process_from_csv(csv_path, alpha_enkf, alpha_eakf)
    
    elif args.csv:
        csv_path = resolve_path(args.csv)
        if not csv_path.exists():
            print(f"CSV 不存在: {csv_path}")
            return 1
        
        print(f"使用 CSV: {csv_path}")
        process_from_csv(csv_path, args.alpha_enkf, args.alpha_eakf)
    
    else:
        print("请指定 --config 或 --csv")
        return 1
    
    print("完成")
    return 0


if __name__ == "__main__":
    exit(main())
