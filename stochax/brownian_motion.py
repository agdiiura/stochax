"""
------------------
brownian_motion.py
------------------

A module for simulation and calibration of brownian motion
stochastic processes
"""

import logging

import numpy as np
import pandas as pd

from scipy.stats import norm

from .core import Bounds, ParameterBound, ABCStochasticProcess

__all__ = ["ArithmeticBrownianMotion", "GeometricBrownianMotion"]

logger = logging.getLogger(__name__)


class ArithmeticBrownianMotion(ABCStochasticProcess):
    r"""
    A class for Arithmetic Brownian Motion

    The stochastic equation is:

    .. math::

        dS_t = \\mu * dt + \\sigma * dB_t

    where :math:`B_t` is the Brownian motion and :math:`S_t` is the process at time :math:`t`.

    Examples:

    .. code-block:: python

        abm = ArithmeticBrownianMotion(mu=0., sigma=0.5)
        paths = abm.simulate(
            initial_value=0.5,
            n_steps=52,
            delta=1/52,
            n_simulations=100
        )

    .. code-block:: python

        data = pd.read_csv('path/to/data.csv')
        abm = ArithmeticBrownianMotion()
        res = abm.calibrate(data)

    """

    _bounds = Bounds(ParameterBound("mu", float, -np.inf, np.inf), ParameterBound("sigma", float, 0.0, np.inf))

    def __init__(self, mu: float | None = None, sigma: float | None = None):
        """
        Initialize the class

        :param mu: drift coefficient
        :param sigma: diffusion coefficient
        """

        self.mu = mu
        self.sigma = sigma

        self._validate_parameters()

    def _simulate(
        self, initial_value: float, n_steps: int, delta: float = 1.0, n_simulations: int = 1, method: str = "exact"
    ) -> pd.DataFrame:
        """
        Simulate Brownian Motion paths

        The time interval of simulation is [0,T] where T = n_steps*delta

        :param initial_value: starting point
        :param n_steps: sample size -> initial value + n observations
        :param delta: sampling interval
        :param n_simulations: number of paths to be simulated
        :param method: simulation method

        :return n_simulations simulations of length n_steps+1
            (included the initial value)
        """

        # use standard normal
        increments = self.mu * delta + self.sigma * np.sqrt(delta) * norm.rvs(size=(n_steps + 1, n_simulations))
        # setting initial condition
        increments[0] = [initial_value] * n_simulations

        return pd.DataFrame(increments.cumsum(axis=0))  # TODO decide index and columns

    def _log_likelihood(self, observations: pd.DataFrame, delta: float = 1.0) -> float:
        """
        Compute the log-likelihood function for ABM process using the parameters
        stored as attributes

        See:

        • PAPER

        :param observations: columns indicates the different
            paths and rows indicates the observations
        :param delta: sampling interval

        :return: the log-likelihood function
        """
        prices = observations.to_numpy()

        p, c = prices[:-1], prices[1:]

        return norm.logpdf(c, loc=p - self.mu * delta, scale=self.sigma * np.sqrt(delta)).sum()

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float, to_array: bool = False
    ) -> dict | np.ndarray:
        """
        Compute th explicit expression for maximum likelihood estimators of an ABM
        process as proposed in

        • Brigo, Damiano, et al.
            "A stochastic processes toolkit for risk management."
            Available at SSRN 1109160 (2007).

        :param observations: columns indicates the different paths
            and rows indicates the observations
        :param delta: sampling interval

        :return: the mle parameters
        """

        returns = observations.diff().dropna().to_numpy()

        # Equal to m = np.nanmean(returns) / delta
        m = (observations.to_numpy()[-1, 0] - observations.to_numpy()[0, 0]) / (len(observations) * delta)
        s = np.nanstd(returns) / np.sqrt(delta)
        if to_array:
            return np.array([m, s])
        else:
            return {
                "mu": m,
                "sigma": s,
            }


class GeometricBrownianMotion(ABCStochasticProcess):
    r"""
    A class for Geometric Brownian Motion

    The stochastic equation is:

    .. math::

        dS_t = S_t(\\mu * dt + \\sigma * dB_t)

    where :math:`B_t` is the Brownian motion and :math:`S_t` is the process at time :math:`t`.

    Examples:

    .. code-block:: python

        gbm = GeometricBrownianMotion(mu=0., sigma=0.5)
        paths = gbm.simulate(
            initial_value=0.5,
            n_steps=52,
            delta=1/52,
            n_simulations=100
        )

    .. code-block:: python

        data = pd.read_csv('path/to/data.csv')
        gbm = GeometricBrownianMotion()
        res = gbm.calibrate(data)

    """

    _bounds = Bounds(ParameterBound("mu", float, -np.inf, np.inf), ParameterBound("sigma", float, 0.0, np.inf))

    def __init__(self, mu: float | None = None, sigma: float | None = None):
        """
        Initialize the class

        :param mu: drift coefficient
        :param sigma: diffusion coefficient
        """

        self.mu = mu
        self.sigma = sigma

        self._validate_parameters()

    def _simulate(
        self, initial_value: float, n_steps: int, delta: float = 1.0, n_simulations: int = 1, method: str = "exact"
    ) -> pd.DataFrame:
        """
        Simulate Brownian Motion paths

        The time interval of simulation is [0,T] where T = n_steps*delta

        :param initial_value: starting point
        :param n_steps: sample size -> initial value + n observations
        :param delta: sampling interval
        :param n_simulations: number of paths to be simulated
        :param method: simulation method

        :return: n_simulations simulations of length n_steps+1
            (included the initial value)
        """

        # use standard normal
        increments = (self.mu - 0.5 * self.sigma**2) * delta + self.sigma * np.sqrt(delta) * norm.rvs(
            size=(n_steps + 1, n_simulations)
        )

        increments = np.exp(increments)
        # setting initial condition
        increments[0] = [initial_value] * n_simulations
        return pd.DataFrame(increments.cumprod(axis=0))  # TODO decide index and columns

    def _log_likelihood(self, observations: pd.DataFrame, delta: float = 1.0) -> float:
        """
        Compute the log-likelihood function for ABM process using the parameters
        stored as attributes

        See:

        • PAPER

        :param observations: columns indicates the different
            paths and rows indicates the observations
        :param delta: sampling interval

        :return: the log-likelihood function
        """
        log_prices = observations.apply(np.log).to_numpy()

        p, c = log_prices[:-1], log_prices[1:]
        return norm.logpdf(c, loc=p - (self.mu - 0.5 * self.sigma**2) * delta, scale=self.sigma * np.sqrt(delta)).sum()

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float, to_array: bool = False
    ) -> dict | np.ndarray:
        """
        Compute th explicit expression for maximum likelihood estimators of an ABM
        process as proposed in

        • Brigo, Damiano, et al.
            "A stochastic processes toolkit for risk management."
            Available at SSRN 1109160 (2007).

        :param observations: columns indicates the different paths
            and rows indicates the observations
        :param delta: sampling interval

        :return: the mle parameters
        """

        returns = observations.apply(np.log).diff().dropna().to_numpy().ravel()

        s = np.nanstd(returns) / np.sqrt(delta)
        m = np.nanmean(returns) / delta + 0.5 * s**2

        if to_array:
            return np.array([m, s])
        else:
            return {
                "mu": m,
                "sigma": s,
            }


if __name__ == "__main__":
    pass
