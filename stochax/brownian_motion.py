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
from numpy.random import Generator

from .core import Bound, Bounds, ABCStochasticProcess

__all__ = ["ArithmeticBrownianMotion", "GeometricBrownianMotion"]

logger = logging.getLogger(__name__)


class ArithmeticBrownianMotion(ABCStochasticProcess):
    r"""
    A class for Arithmetic Brownian Motion

    The stochastic equation is:

    ```math
        dS_t = \mu * dt + \sigma * dB_t
    ```

    where `B_t` is the Brownian motion and `S_t` is the process at time `t`.

    Example:

        >>> abm = ArithmeticBrownianMotion(mu=0., sigma=0.5)
        >>> paths = abm.simulate(
        >>>     initial_value=0.5,
        >>>     n_steps=52,
        >>>     delta=1/52,
        >>>     n_simulations=100
        >>> )

    Example:

        >>> data = pd.read_csv('path/to/data.csv')
        >>> abm = ArithmeticBrownianMotion()
        >>> res = abm.calibrate(data)

    """

    _bounds = Bounds(
        Bound("mu", float, -np.inf, np.inf),
        Bound("sigma", float, 0.0, np.inf),
    )

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        rng: Generator | int | None = None,
    ):
        """
        Initialize the class

        Args:
            mu: drift coefficient
            sigma: diffusion coefficient
            rng: The random state for generating simulations and bootstrap samples

        """

        super().__init__(rng=rng)

        self.mu = mu
        self.sigma = sigma

        self._validate_parameters()

    def _simulate(
        self,
        initial_value: float,
        n_steps: int,
        delta: float = 1.0,
        n_simulations: int = 1,
        method: str = "exact",
    ) -> pd.DataFrame:
        """
        Simulate Brownian Motion paths

        The time interval of simulation is [0,T] where T = n_steps*delta

        Args:
            initial_value: starting point
            n_steps: sample size -> initial value + n observations
            delta: sampling interval
            n_simulations: number of paths to be simulated
            method: simulation method

        Returns:
            n_simulations simulations of length n_steps+1 (included the initial value)

        """

        # use standard normal
        rv = norm()
        rv.random_state = self._rng
        increments = self.mu * delta + self.sigma * np.sqrt(delta) * rv.rvs(
            size=(n_steps + 1, n_simulations)
        )
        # setting initial condition
        increments[0] = [initial_value] * n_simulations

        return pd.DataFrame(increments.cumsum(axis=0))

    def _log_likelihood(self, observations: pd.DataFrame, delta: float = 1.0) -> float:
        """
        Compute the log-likelihood function for ABM process using the parameters
        stored as attributes

        See:

        • PAPER

        Args:
            observations: columns indicates the different
                paths and rows indicates the observations
            delta: sampling interval

        Returns:
            the log-likelihood function

        """
        prices = observations.to_numpy()

        p, c = prices[:-1], prices[1:]

        return norm.logpdf(
            c, loc=p - self.mu * delta, scale=self.sigma * np.sqrt(delta)
        ).sum()

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float
    ) -> dict:
        """
        Compute th explicit expression for maximum likelihood estimators of an ABM
        process as proposed in

        • Brigo, Damiano, et al.
            "A stochastic processes toolkit for risk management."
            Available at SSRN 1109160 (2007).

        Args:
            observations: columns indicates the different paths
                and rows indicates the observations
            delta: sampling interval

        Returns:
            the mle parameters

        """

        prices = observations.to_numpy().ravel()
        returns = np.diff(prices)

        # Equal to m = np.nanmean(returns) / delta
        m = (prices[-1] - prices[0]) / (len(observations) * delta)
        s = np.nanstd(returns) / np.sqrt(delta)
        return {
            "mu": m,
            "sigma": s,
        }


class GeometricBrownianMotion(ABCStochasticProcess):
    r"""
    A class for Geometric Brownian Motion

    The stochastic equation is:

    ```math
        dS_t = S_t(\mu * dt + \sigma * dB_t)
    ```

    where `B_t` is the Brownian motion and `S_t` is the process at time `t`.

    Example:

        >>> gbm = GeometricBrownianMotion(mu=0., sigma=0.5)
        >>> paths = gbm.simulate(
        >>>     initial_value=0.5,
        >>>     n_steps=52,
        >>>     delta=1/52,
        >>>     n_simulations=100
        >>> )

    Example:

        >>> data = pd.read_csv('path/to/data.csv')
        >>> gbm = GeometricBrownianMotion()
        >>> res = gbm.calibrate(data)

    """

    _bounds = Bounds(
        Bound("mu", float, -np.inf, np.inf),
        Bound("sigma", float, 0.0, np.inf),
    )

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        rng: Generator | int | None = None,
    ):
        """
        Initialize the class

        Args:
            mu: drift coefficient
            sigma: diffusion coefficient
            rng: The random state for generating simulations and bootstrap samples

        """

        super().__init__(rng=rng)

        self.mu = mu
        self.sigma = sigma

        self._validate_parameters()

    def _simulate(
        self,
        initial_value: float,
        n_steps: int,
        delta: float = 1.0,
        n_simulations: int = 1,
        method: str = "exact",
    ) -> pd.DataFrame:
        """
        Simulate Geometric Brownian Motion paths

        The time interval of simulation is [0,T] where T = n_steps*delta

        Args:
            initial_value: starting point
            n_steps: sample size -> initial value + n observations
            delta: sampling interval
            n_simulations: number of paths to be simulated
            method: simulation method

        Returns:
            n_simulations simulations of length n_steps+1 (included the initial value)

        """

        # use standard normal
        rv = norm()
        rv.random_state = self._rng
        increments = (self.mu - 0.5 * self.sigma**2) * delta + self.sigma * np.sqrt(
            delta
        ) * rv.rvs(size=(n_steps + 1, n_simulations))

        increments = np.exp(increments)
        # setting initial condition
        increments[0] = [initial_value] * n_simulations
        return pd.DataFrame(increments).cumprod(axis=0)

    def _log_likelihood(self, observations: pd.DataFrame, delta: float = 1.0) -> float:
        """
        Compute the log-likelihood function for GBM process using the parameters
        stored as attributes

        See:

        • PAPER

        Args:
            observations: columns indicates the different
                paths and rows indicates the observations
            delta: sampling interval

        Returns:
            the log-likelihood function

        """
        log_prices = observations.apply(np.log).to_numpy()

        p, c = log_prices[:-1], log_prices[1:]
        return norm.logpdf(
            c,
            loc=p - (self.mu - 0.5 * self.sigma**2) * delta,
            scale=self.sigma * np.sqrt(delta),
        ).sum()

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float
    ) -> dict:
        """
        Compute th explicit expression for maximum likelihood estimators of an ABM
        process as proposed in

        • Brigo, Damiano, et al.
            "A stochastic processes toolkit for risk management."
            Available at SSRN 1109160 (2007).

        Args:
            observations: columns indicates the different paths
                and rows indicates the observations
            delta: sampling interval

        Returns:
            the mle parameters

        """

        prices = np.log(observations.to_numpy().ravel())
        returns = np.diff(prices)

        s = np.nanstd(returns) / np.sqrt(delta)
        # Equal to m = np.nanmean(returns) / delta
        m = (prices[-1] - prices[0]) / (len(observations) * delta)

        return {
            "mu": m + 0.5 * s**2,
            "sigma": s,
        }


if __name__ == "__main__":
    pass
