# Ensemble Data Assimilation and Transformer-Based Optimized Weighted Fusion for State Estimation

Source code for the paper:

> **Ensemble Data Assimilation and Transformer-Based Optimized Weighted Fusion for State Estimation**
>
> Manhong Fan, Yonglong Bai, Lin Ding, Qinghe Yu, Qian Xiao
>
> Submitted to *Computers & Geosciences*

## Overview

This repository provides the implementation of a hybrid state estimation framework that fuses ensemble data assimilation methods (EnKF / EAKF) with deep learning models (Transformer, LSTM, BiLSTM) through an optimized backend linear weighted fusion strategy. All experiments are conducted on the **Lorenz-96** chaotic dynamical system under varying observation sparsity, noise levels, and model error conditions.

## Repository Structure

```
.
├── experiments/                # Experiment orchestration scripts
│   ├── run_pipeline.py         # One-click full pipeline (recommended entry point)
│   ├── run_experiment.py       # Main experiments (baseline / sparse)
│   ├── run_sensitivity.py      # Sensitivity analysis experiments
│   ├── run_alpha_sweep.py      # Fusion weight (alpha) sweep
│   └── run_bilstm_experiment.py# BiLSTM experiment pipeline
│
├── models/                     # Core algorithm implementations
│   ├── lorenz96_model.py       # Lorenz-96 dynamical system, RK4 integrator, observation operator
│   ├── enkf_module.py          # Ensemble Kalman Filter (EnKF)
│   ├── eakf_module.py          # Ensemble Adjustment Kalman Filter (EAKF)
│   ├── transformer.py          # Transformer model (observation-to-state mapping)
│   ├── lstm.py                 # LSTM model
│   └── bilstm.py               # BiLSTM model
│
├── training/                   # Model training scripts
│   ├── train.py                # Train Transformer / LSTM
│   └── train_bilstm.py         # Train BiLSTM
│
├── utils/                      # Data processing and helper scripts
│   ├── prepare_trajectories.py # Generate trajectory files for plotting (Fig.5/8/9)
│   └── make_main_table.py      # Generate summary tables for the paper
│
├── plots/                      # Plotting scripts for paper figures (Fig.2 -- Fig.9)
│   ├── plot_fig2_forest_enhanced.py
│   ├── plot_fig3_sensitivity.py
│   ├── plot_fig4_boxplot.py
│   ├── plot_fig7_dumbbell.py
│   ├── plot_delta_rmse_comparison.py
│   └── ...
│
├── README.md
├── requirements.txt
├── bilstm.py
└── lstm.py
```

## Requirements

- Python >= 3.9
- PyTorch
- NumPy
- Matplotlib
- pandas

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start: Run the Full Pipeline

From the project root directory, run:

```bash
python experiments/run_pipeline.py
```

This will sequentially execute:

1. **Main experiments** (`run_experiment.py`) -- baseline and sparse evaluation + summary
2. **Sensitivity experiments** (`run_sensitivity.py`) -- parameter sensitivity analysis
3. **Trajectory generation** (`prepare_trajectories.py`) -- prepare data for Fig.5/8/9

### Run Individual Steps

#### Main Experiments (Plan A)

```bash
python experiments/run_experiment.py --config baseline --step all
python experiments/run_experiment.py --config sparse   --step all
```

#### Sensitivity Analysis

```bash
python experiments/run_sensitivity.py
```

#### Generate Trajectory Files (for Fig.5/8/9)

```bash
python utils/prepare_trajectories.py --config baseline
python utils/prepare_trajectories.py --config sparse
```

### Generate Paper Figures

After the experiment outputs are ready, generate individual figures:

```bash
python plots/plot_fig2_forest_enhanced.py
python plots/plot_fig3_sensitivity.py
python plots/plot_fig4_boxplot.py
python plots/plot_fig7_dumbbell.py
python plots/plot_delta_rmse_comparison.py
```

## License

This project is provided for academic research purposes.

## Citation

If you find this code useful, please cite:

```bibtex
@article{fan2026ensemble,
  title   = {Ensemble Data Assimilation and Transformer-Based Optimized Weighted Fusion for State Estimation},
  author  = {Fan, Manhong and Bai, Yonglong and Ding, Lin and Yu, Qinghe and Xiao, Qian},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Under review}
}
```
