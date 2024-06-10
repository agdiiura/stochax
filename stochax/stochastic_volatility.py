"""
------------------------
stochastic_volatility.py
------------------------

A module for simulation and calibration of
stochastic processes with stochastic volatility
"""

import logging

from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from numpy.random import Generator

from .core import Bounds, ParameterBound, ABCStochasticProcess

__all__ = ["ConstantElasticityVariance", "Heston"]

logger = logging.getLogger(__name__)


class ConstantElasticityVariance(ABCStochasticProcess):
    """
    Linetsky, Vadim, and Rafael Mendoza.
    "The constant elasticity of variance model."
    Encyclopedia of Quantitative Finance (2010): 328-334.
    """

    pass

    def _log_likelihood(self, *args, **kwargs) -> float:
        pass

    def _simulate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        pass


class Heston(ABCStochasticProcess):
    r"""
    A class for Heston process

    The stochastic system is:

    .. math::

        dS_t = S_t(\mu * dt + \sqrt{V_t} * dB_t)
        dV_t = \kappa * ( \alpha - V_t) * dt + \sigma \sqrt{V_t} * dZ_t
        dB_t dZ_t = \rho dt

    where :math:`B_t` and :math:`Z_t` are the Brownian motion
    and :math:`S_t` is the process at time :math:`t`.

    Examples:

    .. code-block:: python

        heston = Heston()
        pass

    """

    _bounds = Bounds(
        ParameterBound("mu", float, -np.inf, np.inf),
        ParameterBound("kappa", float, 0.0, np.inf),
        ParameterBound("alpha", float),
        ParameterBound("sigma", float, 0.0, np.inf),
        ParameterBound("rho", float, -1.0, +1.0),
    )

    def __init__(
        self,
        mu: float | None = None,
        kappa: float | None = None,
        alpha: float | None = None,
        sigma: float | None = None,
        rho: float | None = None,
        rng: Generator | int | None = None,
    ):
        """
        Initialize the class

        :param mu: drift diffusion coefficient
        :param kappa: volatility mean reversion rate
        :param alpha: volatility long term mean
        :param sigma: volatility coefficient
        :param rho: correlation coefficient
        :param rng: The random state for generating simulations and bootstrap samples
        """

        super().__init__(rng=rng)

        self.mu = mu
        self.kappa = kappa
        self.alpha = alpha
        self.sigma = sigma
        self.rho = rho

        self._validate_parameters()

        # test the feller condition
        if all(par is not None and np.isfinite(par) for par in self.parameters.values()):
            feller_condition = 2.0 * self.alpha * self.kappa >= self.sigma**2
            if not feller_condition:
                raise ValueError(
                    f"The Feller condition (2*kappa*alpha>=sigma^2) = "
                    f"(2*{self.kappa}*{self.alpha}>= {self.sigma}^2) "
                    f"is not verified and with these params process could reach zero"
                )

    def _simulate(
        self,
        initial_value: tuple,
        n_steps: int,
        delta: float = 1.0,
        n_simulations: int = 1,
        method: str = "exact",
    ) -> pd.DataFrame:
        if method not in ["exact", "euler"]:
            raise TypeError("not valid choice for `method`")
        # each column is a simulated path
        i0, v0 = initial_value

        observations = i0 * np.ones((n_steps + 1, n_simulations))

        vol = v0 * np.ones((n_steps + 1, n_simulations))
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        Z = multivariate_normal.rvs(
            mean=[0.0, 0.0], cov=cov, size=(n_steps + 1, n_simulations)
        )

        if method == "euler":
            for i in range(1, n_steps + 1):
                v_prev = vol[i - 1]
                observations[i] = observations[i - 1] * np.exp(
                    (self.mu - 0.5 * v_prev) * delta
                    + np.sqrt(v_prev * delta) * Z[i - 1, :, 0]
                )
                vol[i] = np.maximum(
                    v_prev
                    + self.kappa * (self.alpha - v_prev) * delta
                    + self.sigma * np.sqrt(v_prev * delta) * Z[i - 1, :, 1],
                    0,
                )

        elif method == "exact":
            raise NotImplementedError

        return pd.DataFrame(observations)

    def _log_likelihood(self, *args, **kwargs) -> float:
        pass

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float
    ) -> dict:
        """
        Compute th explicit expression for maximum likelihood estimators of Heston
        process as proposed in

        • Atiya, Amir F., and Steve Wall.
            "An analytic approximation of the likelihood function
            for the Heston model volatility estimation problem."
            Quantitative Finance 9.3 (2009): 289-296.

        :param observations: columns indicates the path and rows indicates the observations
        :param delta: sampling interval

        :return: parameters: mle parameters
        """
        pass

    def _extended_kalman_filter(self):
        """
        Javaheri, Alireza, Delphine Lautier, and Alain Galli.
        "Filtering in finance."
        Wilmott 3 (2003): 67-83.

        Wang, Ximei, et al.
        "Parameter estimates of Heston stochastic volatility model
        with MLE and consistent EKF algorithm."
        Science China Information Sciences 61 (2018): 1-17.
        """
        pass
