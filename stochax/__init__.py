"""
-------
stochax
-------

A python package for the simulation and calibration of
stochastic processes

Examples:

    >>> from stochax import ArithmeticBrownianMotion
    >>> abm = ArithmeticBrownianMotion(mu=0, sigma=0.5)
    >>> paths = abm.simulate(
    >>>     initial_value=0.5,
    >>>     n_steps=52,
    >>>     delta=1/52,
    >>>     n_simulations=100
    >>> )


    >>> import pandas as pd
    >>> from stochax import GeometricBrownianMotion
    >>>
    >>> data = pd.read_csv('path/to/data.csv')
    >>> gbm = GeometricBrownianMotion()
    >>> res = gbm.calibrate(data)
    >>>
    >>> print(res.get_summary())

"""

from pathlib import Path
from importlib.metadata import version

from .core import *
from .mean_reverting import *
from .brownian_motion import *
from .calibration_results import *


def _read_version() -> str:
    """Read version from metadata or pyproject.toml"""

    try:
        return version("stochax")

    except Exception:  # pragma: no cover
        # For development
        file = Path(__file__).absolute().parents[1] / "pyproject.toml"

        if file.exists():
            with open(file, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    if line.startswith("version"):
                        return line.split("=")[-1].strip()

        return "0.x"


__version__ = _read_version()

__all__ = [itm for itm in dir() if not itm.startswith("_")]
