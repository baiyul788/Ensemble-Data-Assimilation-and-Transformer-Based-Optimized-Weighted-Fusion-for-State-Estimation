import argparse
import subprocess
import sys
from pathlib import Path


def _slug_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def _run(cmd: list[str], repo_root: Path, desc: str) -> None:
    print("\n" + "=" * 80)
    print(desc)
    print("  " + " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=repo_root, check=True)


def _check_exists(path: Path, title: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {title}: {path}")


def _check_glob_nonempty(pattern: str, root: Path, title: str) -> None:
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing required outputs: {title}: pattern={pattern}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-click pipeline: regenerate experiment CSV/NPZ outputs for plots "
            "(Fig.2/3/4/6/7 require CSVs; Fig.5/8/9 require trajectories)."
        )
    )
    parser.add_argument("--skip_main", action="store_true", help="Skip main experiments (run_experiment)")
    parser.add_argument("--skip_sensitivity", action="store_true", help="Skip sensitivity (run_sensitivity)")
    parser.add_argument("--skip_trajectories", action="store_true", help="Skip trajectories (prepare_trajectories)")
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Do not run run_experiment summary step (evaluate only)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result_dir = repo_root / "result"

    # Consistent with run_experiment.py
    total_time = 20.0
    dt_obs = 0.2
    window = 5
    n_seeds = 5

    # Final alphas
    sparse_aen, sparse_aea = 0.20, 0.25
    baseline_aen, baseline_aea = 0.33, 0.44

    if not args.skip_main:
        _run([sys.executable, "experiments/run_experiment.py", "--config", "sparse", "--step", "evaluate"], repo_root, "Main experiment: sparse evaluate")
        if not args.no_summary:
            _run([sys.executable, "experiments/run_experiment.py", "--config", "sparse", "--step", "summary"], repo_root, "Main experiment: sparse summary")

        _run([sys.executable, "experiments/run_experiment.py", "--config", "baseline", "--step", "evaluate"], repo_root, "Main experiment: baseline evaluate")
        if not args.no_summary:
            _run([sys.executable, "experiments/run_experiment.py", "--config", "baseline", "--step", "summary"], repo_root, "Main experiment: baseline summary")

    if not args.skip_sensitivity:
        _run([sys.executable, "experiments/run_sensitivity.py"], repo_root, "Sensitivity analysis (sparse baseline, final alphas)")

    if not args.skip_trajectories:
        _run([sys.executable, "utils/prepare_trajectories.py", "--config", "sparse"], repo_root, "Prepare trajectories: sparse")
        _run([sys.executable, "utils/prepare_trajectories.py", "--config", "baseline"], repo_root, "Prepare trajectories: baseline")

    # ------------------------------------------------------------------
    # Validate key outputs for plots
    # ------------------------------------------------------------------
    expected_fig2_sparse = result_dir / (
        f"plan_a_multiseed_s6_N10_"
        f"sigObs{_slug_float(2.0)}_sigModel{_slug_float(0.1)}_"
        f"T{_slug_float(total_time)}_dtobs{_slug_float(dt_obs)}_"
        f"w{window}_aEn{_slug_float(sparse_aen)}_aEa{_slug_float(sparse_aea)}_"
        f"seeds{n_seeds}.csv"
    )
    expected_fig2_baseline = result_dir / (
        f"plan_a_multiseed_s9_N20_"
        f"sigObs{_slug_float(2.0)}_sigModel{_slug_float(0.0)}_"
        f"T{_slug_float(total_time)}_dtobs{_slug_float(dt_obs)}_"
        f"w{window}_aEn{_slug_float(baseline_aen)}_aEa{_slug_float(baseline_aea)}_"
        f"seeds{n_seeds}.csv"
    )

    if not args.skip_main:
        _check_exists(expected_fig2_sparse, "Fig.2 sparse plan_a_multiseed")
        _check_exists(expected_fig2_baseline, "Fig.2 baseline plan_a_multiseed")

    if not args.skip_sensitivity:
        _check_exists(result_dir / "sensitivity_summary.csv", "Fig.3/6/7 sensitivity_summary.csv")
        _check_glob_nonempty(
            f"sensitivity_sens_*_w{window}_aEn{_slug_float(sparse_aen)}_aEa{_slug_float(sparse_aea)}.csv",
            result_dir,
            "Fig.4 sensitivity_sens_*.csv",
        )

    if not args.skip_trajectories:
        _check_exists(
            result_dir
            / "trajectories_dataset_l96_n36_dt0p01_T20_dtobs0p2_s6_N10_sigObs2_sigModel0p1_seed0_sparse.npz",
            "Fig.5/8/9 trajectories (sparse seed0)",
        )
        _check_exists(
            result_dir
            / "trajectories_dataset_l96_n36_dt0p01_T20_dtobs0p2_s9_N20_sigObs2_sigModel0_seed0_baseline.npz",
            "Trajectories (baseline seed0)",
        )

    print("\nAll requested pipeline steps finished and key outputs are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
