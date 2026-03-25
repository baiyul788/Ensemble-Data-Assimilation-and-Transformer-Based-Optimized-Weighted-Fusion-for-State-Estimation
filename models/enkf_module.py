"""enkf_module.py

Ensemble Kalman Filter (EnKF) utilities for Lorenz-96.

This module implements the classic *perturbed-observation* EnKF analysis update
in the style of Burgers et al. (1998).

References (publicly findable):
- Evensen, G. (1994). Sequential data assimilation with a nonlinear
  quasi-geostrophic model using Monte Carlo methods to forecast error statistics.
  *J. Geophys. Res.*, 99(C5), 10143–10162. DOI: 10.1029/94JC00572
  https://agupubs.onlinelibrary.wiley.com/doi/10.1029/94JC00572
- Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Mon. Wea. Rev.*, 126(6), 1719–1724.
  DOI: 10.1175/1520-0493(1998)126<1719:ASITEK>2.0.CO;2
  https://journals.ametsoc.org/view/journals/mwre/126/6/1520-0493_1998_126_1719_asitek_2.0.co_2.xml
- Houtekamer, P. L., & Mitchell, H. L. (1998). Data assimilation using an
  ensemble Kalman filter technique. *Mon. Wea. Rev.*, 126(3), 796–811.
  DOI: 10.1175/1520-0493(1998)126<0796:DAUAEK>2.0.CO;2
  https://journals.ametsoc.org/view/journals/mwre/126/3/1520-0493_1998_126_0796_dauaek_2.0.co_2.xml
- Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation and
  practical implementation. *Ocean Dynamics*, 53, 343–367.
  DOI: 10.1007/s10236-003-0036-9
  https://link.springer.com/article/10.1007/s10236-003-0036-9
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .lorenz96_model import F, Lorenz96, RK4, dt, h, n


def _cyclic_distance(i: np.ndarray, j: np.ndarray, n_state: int) -> np.ndarray:
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    d = np.abs(i[..., None] - j[None, ...])
    return np.minimum(d, int(n_state) - d)


def _gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """Gaspari-Cohn compactly supported correlation function.

    Args:
        r: normalized distance d / L, where L is localization radius.
    """
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


def _localization_matrices(config, n_state: int, m_obs: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build localization matrices for P_xy (n_state x m_obs) and P_yy (m_obs x m_obs)."""
    if config is None or (not hasattr(config, "localization_radius")) or (config.localization_radius is None):
        return None, None
    L = float(config.localization_radius)
    if not np.isfinite(L) or L <= 0.0:
        return None, None

    obs_idx = None
    if hasattr(config, "obs_indices") and (config.obs_indices is not None):
        obs_idx = np.asarray(config.obs_indices, dtype=int).reshape(-1)
    if obs_idx is None or obs_idx.size != int(m_obs):
        obs_idx = np.arange(int(m_obs), dtype=int)

    state_idx = np.arange(int(n_state), dtype=int)
    d_xy = _cyclic_distance(state_idx, obs_idx, n_state=int(n_state))
    C_xy = _gaspari_cohn(d_xy / L)

    d_yy = _cyclic_distance(obs_idx, obs_idx, n_state=int(n_state))
    C_yy = _gaspari_cohn(d_yy / L)
    return C_xy, C_yy


def enkf_analysis(
    x_f: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    obs_op,
    rng: np.random.Generator,
    config=None,
) -> np.ndarray:
    """Perturbed-observation EnKF analysis update.

    Args:
        x_f: forecast ensemble [n, N]
        y: observation vector [m]
        R: observation error covariance [m, m]
        obs_op: callable mapping x -> y (vector)
        rng: numpy RNG
        config: optional configuration object with fields:
            - inflation_factor: optional multiplicative inflation factor
            - localization_radius: optional localization radius
            - obs_indices: optional observation indices

    Returns:
        analysis ensemble [n, N]
    """
    x_f = np.asarray(x_f, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_state, n_ens = x_f.shape
    m = y.shape[0]

    R = np.asarray(R, dtype=float)
    if R.ndim == 0:
        Rm = float(R) * np.eye(m)
    elif R.ndim == 1:
        Rm = np.diag(R)
    else:
        Rm = R

    y_f = np.column_stack([np.asarray(obs_op(x_f[:, i]), dtype=float).reshape(-1) for i in range(n_ens)])

    x_mean = np.mean(x_f, axis=1, keepdims=True)
    y_mean = np.mean(y_f, axis=1, keepdims=True)
    X = x_f - x_mean
    Y = y_f - y_mean

    scale = 1.0 / np.sqrt(n_ens - 1)
    Xs = X * scale
    Ys = Y * scale

    P_xy = Xs @ Ys.T
    P_yy = Ys @ Ys.T

    # Optional localization (taper covariances)
    C_xy, C_yy = _localization_matrices(config, n_state=n_state, m_obs=int(y.shape[0]))
    if C_xy is not None:
        P_xy = P_xy * C_xy

    P_yy = P_yy + Rm

    K = np.linalg.solve(P_yy, P_xy.T).T

    if R.ndim == 0:
        pert = rng.normal(0.0, np.sqrt(float(R)), size=(m, n_ens))
    elif R.ndim == 1:
        pert = rng.normal(0.0, np.sqrt(R).reshape(-1, 1), size=(m, n_ens))
    else:
        pert = rng.multivariate_normal(np.zeros(m), Rm, size=n_ens).T

    innov = (y.reshape(-1, 1) + pert) - y_f

    return x_f + K @ innov


def run_experiment(
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
    Run a complete EnKF baseline experiment
    
    Parameters:
    - x0b: initial background field [36]
    - yo: observation data [s, nt_m]
    - ind_m: observation time index
    - nt: integration step number
    - nt_m: assimilation number
    - N: ensemble member number
    - Q: model error covariance matrix
    - R: observation error covariance matrix
    - sig_b: initial background error standard deviation
    
    Returns:
    - xa: analysis field [36, nt+1]
    - xai: ensemble analysis field [36, N]
    """
    if verbose:
        print("Start EnKF assimilation...")
        print(f"Ensemble member number: {N}")
        print(f"Observation number: {nt_m}")
    rng = np.random.default_rng() if rng is None else rng

    x0b = np.asarray(x0b, dtype=float).reshape(n)
    xai = x0b.reshape(-1, 1) + rng.normal(0.0, float(sig_b), size=(n, N))
    
    # EnKF assimilation
    xa = np.zeros([n, nt+1])
    xa[:, 0] = np.mean(xai, 1)
    km = 0
    
    for k in range(nt):
        Q_arr = np.asarray(Q, dtype=float)
        if Q_arr.ndim == 0:
            model_noise = rng.normal(0.0, np.sqrt(float(Q_arr)), size=(n, N))
        elif Q_arr.ndim == 1:
            model_noise = rng.normal(0.0, np.sqrt(Q_arr).reshape(-1, 1), size=(n, N))
        else:
            model_noise = rng.multivariate_normal(np.zeros(n), Q_arr, size=N).T

        for i in range(N):
            xai[:, i] = RK4(Lorenz96, xai[:, i], dt, F) + model_noise[:, i]
        
        xa[:, k+1] = np.mean(xai, 1)
        
        if (km < nt_m) and (k + 1 == ind_m[km]):
            obs_op = (lambda x: h(x, config)) if config is not None else (lambda x: h(x))

            # Optional multiplicative inflation (applied to forecast ensemble before analysis)
            if config is not None and hasattr(config, "inflation_factor") and (config.inflation_factor is not None):
                lam = float(config.inflation_factor)
                if np.isfinite(lam) and lam > 0.0 and (abs(lam - 1.0) > 1e-12):
                    xb = np.mean(xai, axis=1, keepdims=True)
                    A = xai - xb
                    xai = xb + np.sqrt(lam) * A

            xai = enkf_analysis(xai, yo[:, km], R, obs_op, rng, config)
            xa[:, k+1] = np.mean(xai, 1)
            km = km + 1
            
            if verbose and (km % 10 == 0):  
                print(f"Completed {km}/{nt_m} assimilations")

    if verbose:
        print("EnKF assimilation completed!")
    
    return xa, xai