"""eakf_module.py

Ensemble Adjustment Kalman Filter (EAKF) utilities for Lorenz-96.

This module implements a *serial deterministic* ensemble adjustment update for
independent (diagonal) observation errors. The update follows the ensemble
square-root filter / EAKF family (no perturbed observations).

References (publicly findable):
- Anderson, J. L. (2001). An ensemble adjustment Kalman filter for data
  assimilation. *Mon. Wea. Rev.*, 129(12), 2884–2903.
  DOI: 10.1175/1520-0493(2001)129<2884:AEAKFF>2.0.CO;2
  https://journals.ametsoc.org/view/journals/mwre/129/12/1520-0493_2001_129_2884_aeakff_2.0.co_2.xml
- Whitaker, J. S., & Hamill, T. M. (2002). Ensemble data assimilation without
  perturbed observations. *Mon. Wea. Rev.*, 130(7), 1913–1924.
  DOI: 10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2
  https://journals.ametsoc.org/view/journals/mwre/130/7/1520-0493_2002_130_1913_edawpo_2.0.co_2.xml
- Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., & Whitaker, J. S.
  (2003). Ensemble square root filters. *Mon. Wea. Rev.*, 131(7), 1485–1490.
  DOI: 10.1175/1520-0493(2003)131<1485:ESRF>2.0.CO;2
  https://journals.ametsoc.org/view/journals/mwre/131/7/1520-0493_2003_131_1485_esrf_2.0.co_2.xml
- Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation and
  practical implementation. *Ocean Dynamics*, 53, 343–367.
  DOI: 10.1007/s10236-003-0036-9
  https://link.springer.com/article/10.1007/s10236-003-0036-9
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .lorenz96_model import F, Lorenz96, RK4, dt, h, n


def _cyclic_distance(i: np.ndarray, j: int, n_state: int) -> np.ndarray:
    i = np.asarray(i, dtype=int)
    d = np.abs(i - int(j))
    return np.minimum(d, int(n_state) - d)


def _gaspari_cohn(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    out = np.zeros_like(a, dtype=float)

    m1 = a <= 1.0
    x = a[m1]
    out[m1] = (
        1.0
        - (5.0 / 3.0) * x**2
        + (5.0 / 8.0) * x**3
        + 0.5 * x**4
        - 0.25 * x**5
    )

    m2 = (a > 1.0) & (a <= 2.0)
    x = a[m2]
    out[m2] = (
        (4.0 - 5.0 * x + (5.0 / 3.0) * x**2 + (5.0 / 8.0) * x**3 - 0.5 * x**4 + (1.0 / 12.0) * x**5)
        / x
    )

    out[a > 2.0] = 0.0
    return out


def eakf_analysis_serial(
    x_f: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    obs_op,
    config=None,
) -> np.ndarray:
    """Serial deterministic EAKF/EnSRF-style analysis update.

    Notes:
        - Assumes observation errors are independent; if R is full, only diag(R)
          is used.
        - obs_op may be nonlinear; we use the ensemble of predicted observations.
    """
    x_a = np.asarray(x_f, dtype=float).copy()
    _, n_ens = x_a.shape
    y = np.asarray(y, dtype=float).reshape(-1)
    m = y.shape[0]

    R = np.asarray(R, dtype=float)
    if R.ndim == 0:
        rdiag = np.full(m, float(R))
    elif R.ndim == 1:
        rdiag = R.reshape(-1)
    else:
        rdiag = np.diag(R)

    # Optional localization settings
    loc_radius = None
    obs_indices = None
    if hasattr(config, 'localization_radius'):
        loc_radius = config.localization_radius
    if hasattr(config, 'obs_indices'):
        obs_indices = config.obs_indices

    for j in range(m):
        yj = float(y[j])
        rj = float(rdiag[j])

        yj_ens = np.asarray([
            np.asarray(obs_op(x_a[:, i]), dtype=float).reshape(-1)[j]
            for i in range(n_ens)
        ])

        yj_mean = float(np.mean(yj_ens))
        yj_pert = yj_ens - yj_mean
        var_y = float(np.dot(yj_pert, yj_pert) / (n_ens - 1))
        if not np.isfinite(var_y) or var_y <= 0.0:
            continue

        x_mean = np.mean(x_a, axis=1, keepdims=True)
        X = x_a - x_mean
        cov_xy = (X @ yj_pert.reshape(-1, 1)) / (n_ens - 1)

        # Optional localization: taper cov_xy by Gaspari-Cohn using cyclic distance
        if loc_radius is not None and obs_indices is not None:
            dist = _cyclic_distance(np.arange(n), obs_indices[j], n)
            taper = _gaspari_cohn(dist / loc_radius)
            cov_xy *= taper.reshape(-1, 1)

        kg = var_y / (var_y + rj)
        x_mean_a = x_mean + (cov_xy / (var_y + rj)) * (yj - yj_mean)

        alpha = np.sqrt(max(0.0, 1.0 - kg))
        yj_pert_a = alpha * yj_pert

        X_a = X + (cov_xy / var_y) @ (yj_pert_a - yj_pert).reshape(1, -1)
        X_a = X_a - np.mean(X_a, axis=1, keepdims=True)
        x_a = x_mean_a + X_a

    return x_a


def run_eakf_experiment(
    x0b,
    yo,
    ind_m,
    nt,
    nt_m,
    N,
    Q,
    R,
    sig_b,
    verbose: bool = False,
    config=None,
    rng: Optional[np.random.Generator] = None,
):
    """
    Run a complete EAKF experiment
    
    Parameters:
    - x0b: Initial background state [36]
    - yo: Observation data [s, nt_m]
    - ind_m: Observation time indices
    - nt: Integration steps
    - nt_m: Assimilation times
    - N: Ensemble size
    - Q: Model error covariance matrix
    - R: Observation error covariance matrix
    - sig_b: Initial background error standard deviation
    
    Returns:
    - xa: Analysis state [36, nt+1]
    - xai: Ensemble analysis state [36, N]
    """
    if verbose:
        print("Starting EAKF assimilation...")
        print(f"Ensemble size: {N}, Assimilation times: {nt_m}")
    rng = np.random.default_rng() if rng is None else rng

    x0b = np.asarray(x0b, dtype=float).reshape(n)
    xai = x0b.reshape(-1, 1) + rng.normal(0.0, float(sig_b), size=(n, N))
    
    xa = np.zeros([n, nt+1])
    xa[:, 0] = np.mean(xai, 1)
    
    obs_op = (lambda x: h(x, config)) if config is not None else (lambda x: h(x))

    # Integration and assimilation loop
    km = 0
    for k in range(nt):
        Q_arr = np.asarray(Q, dtype=float)
        if Q_arr.ndim == 0:
            model_noise = rng.normal(0.0, np.sqrt(float(Q_arr)), size=(n, N))
        elif Q_arr.ndim == 1:
            model_noise = rng.normal(0.0, np.sqrt(Q_arr).reshape(-1, 1), size=(n, N))
        else:
            model_noise = rng.multivariate_normal(np.zeros(n), Q_arr, size=N).T

        # Forecast
        for i in range(N):
            xai[:, i] = RK4(Lorenz96, xai[:, i], dt, F) + model_noise[:, i]
        
        # Save forecast mean
        xa[:, k+1] = np.mean(xai, 1)
        
        # Assimilate at observation times
        if (km < nt_m) and (k + 1 == ind_m[km]):
            # Optional multiplicative inflation
            if hasattr(config, 'inflation_factor') and (config.inflation_factor is not None):
                xb = np.mean(xai, axis=1, keepdims=True)
                A = xai - xb
                xai = xb + np.sqrt(config.inflation_factor) * A

            xai = eakf_analysis_serial(xai, yo[:, km], R, obs_op, config=config)
            xa[:, k+1] = np.mean(xai, 1)
            km += 1
            
            if verbose and (km % 10 == 0):
                print(f"Completed {km}/{nt_m} assimilations")
    
    if verbose:
        print("EAKF assimilation completed!")
    
    return xa, xai