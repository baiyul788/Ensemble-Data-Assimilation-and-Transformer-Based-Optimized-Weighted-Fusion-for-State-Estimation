"""lorenz96_model.py

Lorenz-96 single-scale model utilities: model RHS, RK4 integrator, and a simple
linear observation operator (subsampling).

References (publicly findable):
- Lorenz, E. N. (1996). Predictability: A problem partly solved.
  In *Predictability* (ECMWF Seminar Proceedings), Vol. 1, 1–18.
  (Often cited as the original source of the Lorenz-96 model.)
  PDF (ECMWF): https://www.ecmwf.int/sites/default/files/elibrary/1995/10829-predictability-problem-partly-solved.pdf
- Lorenz, E. N., & Emanuel, K. A. (1998). Optimal sites for supplementary weather
  observations: Simulation with a small model. *J. Atmos. Sci.*, 55(3), 399–414.
  DOI: 10.1175/1520-0469(1998)055<0399:OSFSWO>2.0.CO;2
  https://journals.ametsoc.org/view/journals/atsc/55/3/1520-0469_1998_055_0399_osfswo_2.0.co_2.xml
- Lorenz, E. N. (2005). Designing chaotic models. *J. Atmos. Sci.*, 62(5), 1574–1587.
  DOI: 10.1175/JAS3430.1
  https://journals.ametsoc.org/view/journals/atsc/62/5/jas3430.1.xml
- Kerin, J., & Engler, H. (2020). On the Lorenz '96 model and some generalizations.
  (Open-access; contains the standard Lorenz-96 equation explicitly.)
  arXiv: https://arxiv.org/abs/2005.07767
  PDF: https://arxiv.org/pdf/2005.07767.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

def _default_obs_indices(n_state: int, m: int = 9) -> np.ndarray:
    di = int(n_state / m)
    return np.asarray([(i + 1) * di - 1 for i in range(m)], dtype=int)


@dataclass(frozen=True)
class Lorenz96Config:
    n: int = 36
    F: float = 8.0
    dt: float = 0.01
    obs_indices: Optional[np.ndarray] = None
    localization_radius: Optional[float] = None
    inflation_factor: Optional[float] = None

    @property
    def s(self) -> int:
        idx = self.obs_indices if self.obs_indices is not None else _default_obs_indices(self.n)
        return int(np.asarray(idx).size)

    def resolved_obs_indices(self) -> np.ndarray:
        return self.obs_indices if self.obs_indices is not None else _default_obs_indices(self.n)


def Lorenz96(state: np.ndarray, *args) -> np.ndarray:
    """Lorenz-96 single-scale RHS.

    d x_i / dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

    Periodic boundary conditions are implemented via np.roll.
    """
    x = np.asarray(state, dtype=float)
    F_in = float(args[0])
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F_in

def RK4(rhs, state, dt, *args, clip: Optional[float] = 50.0) -> np.ndarray:
    """4th-order Runge–Kutta integrator (optionally clipped for robustness)."""

    def _sanitize(z: np.ndarray) -> np.ndarray:
        z = np.nan_to_num(z, nan=0.0, posinf=np.inf, neginf=-np.inf)
        if clip is None:
            return z
        return np.clip(z, -clip, clip)

    s1 = _sanitize(np.asarray(state, dtype=float))
    k1 = rhs(s1, *args)
    k2 = rhs(_sanitize(s1 + 0.5 * dt * k1), *args)
    k3 = rhs(_sanitize(s1 + 0.5 * dt * k2), *args)
    k4 = rhs(_sanitize(s1 + dt * k3), *args)
    return _sanitize(s1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

def h(x, config=None):
    """
    Observation operator: extract observations based on configuration

    Args:
        x: state vector [n]
        config: optional ExperimentConfig, corresponding to obs_indices and s

    Returns:
        observation vector [s]
    """
    x = np.asarray(x)
    if config is not None and hasattr(config, 'obs_indices'):
        return x[np.asarray(config.obs_indices, dtype=int)]
    return x[_default_obs_indices(x.shape[0], m=9)]

# Lorenz96 parameters (backward compatibility)
n = 36
F = 8
dt = 0.01