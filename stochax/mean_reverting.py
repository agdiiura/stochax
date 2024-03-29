"""
-----------------
mean_reverting.py
-----------------

A module for simulation and calibration of mean reverting
stochastic processes
"""

import abc
import logging

import numpy as np
import pandas as pd

from scipy.stats import ncx2, norm, gamma

from .core import Bounds, ParameterBound, ABCStochasticProcess

__all__ = ["OrnsteinUhlenbeck", "CoxIngersollRoss"]

logger = logging.getLogger(__name__)


class ABCMeanReverting(ABCStochasticProcess, abc.ABC):
    def stationary_distribution(self, n_simulations: int = 1) -> np.ndarray:
        """
        Generate n_simulations values from the stationary distribution of an OU process

        :params n_simulations: number of values to be simulated
        :return: the stationarity distribution

        """
        self._validate_parameters()
        return self._stationary_distribution(n_simulations=n_simulations)

    @abc.abstractmethod
    def _stationary_distribution(self, n_simulations: int = 1) -> np.ndarray:
        pass


class OrnsteinUhlenbeck(ABCMeanReverting):
    r"""
    The Ornstein-Uhlenbeck process class:

    The stochastic equation is:

    .. math::

        dS_t = \\kappa * ( \\alpha - S_t) * dt + \\sigma * dB_t

    where :math:`B_t` is the Brownian motion and :math:`S_t` is the process at time :math:`t`.

    Examples:

    .. code-block:: python

        ou = OrnsteinUhlenbeck(kappa=1., theta=0., sigma=0.5)
        paths = ou.simulate(
            initial_value=0.5,
            n_steps=52,
            delta=1/52,
            n_simulations=100
        )

    .. code-block:: python

        data = pd.read_csv('path/to/data.csv')
        ou = OrnsteinUhlenbeck()
        res = ou.calibrate(data)

    """

    _bounds = Bounds(
        ParameterBound("kappa", float, 0.0, np.inf),
        ParameterBound("alpha", float),
        ParameterBound("sigma", float, 0.0, np.inf),
    )

    def __init__(self, kappa: float | None = None, alpha: float | None = None, sigma: float | None = None):
        """
        Initialize the class

        :param kappa: mean reversion rate
        :param alpha: long term mean
        :param sigma: volatility coefficient
        """

        self.kappa = kappa
        self.alpha = alpha
        self.sigma = sigma

        self._validate_parameters()

    def _stationary_distribution(self, n_simulations: int = 1) -> np.ndarray:
        """
        Generate n_simulations values from the stationary distribution of an OU process

        :params n_simulations: number of values to be simulated
        :return: the stationarity distribution
        """
        return norm.rvs(loc=self.alpha, scale=self.sigma * np.sqrt(1.0 / (2.0 * self.kappa)), size=n_simulations)

    def _simulate(
        self, initial_value: float, n_steps: int, delta: float = 1.0, n_simulations: int = 1, method: str = "exact"
    ) -> pd.DataFrame:
        """
        Simulate Ornstein Uhlenbeck paths

        The time interval of simulation is [0,T] where T = n_steps*delta

        :param initial_value: starting point
        :param n_steps: sample size -> initial value + n observations
        :param delta: sampling interval
        :param n_simulations: number of paths to be simulated
        :param method: simulation method

        :return simulation: n_simulations simulations of length n_steps+1
            (included the initial value)
        """

        # each column will be a simulated path
        simulations = initial_value * np.ones((n_steps + 1, n_simulations))

        e = np.exp(-self.kappa * delta)
        # use standard normal
        brownian = self.sigma * np.sqrt((1.0 - e**2) / (2 * self.kappa)) * norm.rvs(size=(n_steps + 1, n_simulations))

        for i, b in enumerate(brownian[1:]):
            # in each iteration simulate one step of EACH path
            simulations[i + 1, :] = simulations[i, :] * e + self.alpha * (1.0 - e) + b

        return pd.DataFrame(simulations)  # TODO decide index and columns

    def _log_likelihood(self, observations: pd.DataFrame, delta: float = 1.0) -> float:
        """
        Compute the log-likelihood function for Ornstein-Uhlenbeck process using the parameters
        stored as attributes

        See:

        • Franco, José Carlos Garcıa.
            "Maximum likelihood estimation of mean reverting processes."
            Real Options Practice (2003).

        • Holý, Vladimír, and Petra Tomanová.
            "Estimation of Ornstein-Uhlenbeck Process Using Ultra-High-Frequency
            Data with Application to Intraday Pairs Trading Strategy."
            arXiv preprint arXiv:1811.09312 (2018).

        :param observations: columns indicates the different
            paths and rows indicates the observations
        :param delta: sampling interval
        :return value: maximum of a log-likelihood function
        """
        prices = observations.to_numpy().ravel()

        p, c = prices[:-1], prices[1:]
        e = np.exp(-self.kappa * delta)

        return norm.logpdf(
            c, loc=self.alpha + (p - self.alpha) * e, scale=0.5 * self.sigma**2 / self.kappa * (1.0 - e**2)
        ).sum()

    def _maximum_likelihood_estimation(
        self, observations: pd.DataFrame, delta: float, to_array: bool = False
    ) -> dict | np.ndarray:
        """
        Compute th explicit expression for maximum likelihood estimators of an Ornstein-Uhlenbeck
        process as proposed in

        • Tang, Cheng Yong, and Song Xi Chen.
            "Parameter estimation and bias correction for diffusion processes."
            Journal of Econometrics 149.1 (2009): 65-81.

        :param observations: columns indicates the different paths
            and rows indicates the observations
        :param delta: sampling interval
        :param to_array: option to return an array instead of a dict

        :return: parameters: mle parameters
        """

        n = len(observations) - 1

        observations = observations.to_numpy()
        obs_x = observations[0:n, :].sum(axis=0)[0]
        obs_y = observations[1 : n + 1, :].sum(axis=0)[0]
        obs_xx = (observations[0:n, :] ** 2).sum(axis=0)[0]
        obs_xy = (observations[0:n, :] * observations[1 : n + 1, :]).sum(axis=0)[0]
        obs_yy = (observations[1 : n + 1, :] ** 2).sum(axis=0)[0]

        n_inv = 1.0 / n
        # below the explicit expression for the parameters (alpha,kappa,sigma)
        b1 = (n_inv * obs_xy - (n_inv**2) * obs_y * obs_x) / (n_inv * obs_xx - (n_inv * obs_x) ** 2)
        b1_minus = 1.0 - b1
        b2 = n_inv * (obs_y - obs_x * b1) / b1_minus
        b3 = n_inv * (
            obs_yy
            + (b1**2) * obs_xx
            + (b2**2) * (b1_minus**2) * n
            - 2 * obs_xy * b1
            - 2 * obs_y * b1_minus * b2
            + 2 * b1 * b2 * b1_minus * obs_x
        )

        # condition on b1 to get positive kappa, condition on b3 to get
        # sigma > 0 (delta is strictly positive)
        if 0.0 < b1 < 1.0 and b3 > 0.0:
            kappa = -(1.0 / delta) * np.log(b1)
            alpha = b2
            sigma = np.sqrt(2.0 * kappa * b3 / (1.0 - b1**2))

        else:
            kappa = np.nan
            alpha = np.nan
            sigma = np.nan

        if to_array:
            return np.array([kappa, alpha, sigma])
        else:
            return {
                "kappa": kappa,
                "alpha": alpha,
                "sigma": sigma,
            }


class CoxIngersollRoss(ABCMeanReverting):
    r"""
    The CIR (Cox-Ingersoll-Ross) process class:

    The stochastic equation is:

    .. math::

        dS_t = \\kappa * ( \\alpha - S_t) * dt + \\sigma \\sqrt{S_t} * dB_t

    where :math:`B_t` is the Brownian motion and :math:`S_t` is the process at time :math:`t`.

    Examples:

    .. code-block:: python

        cir = CoxIngersollRoss(kappa=1., theta=0., sigma=0.5)
        paths = cir.simulate(
            initial_value=0.5,
            n_steps=52,
            delta=1/52,
            n_simulations=100
        )

    .. code-block:: python

        data = pd.read_csv('path/to/data.csv')
        cir = CoxIngersollRoss()
        res = cir.calibrate(data)

    """

    _bounds = Bounds(
        ParameterBound("kappa", float, 0.0, np.inf),
        ParameterBound("alpha", float),
        ParameterBound("sigma", float, 0.0, np.inf),
    )

    def __init__(self, kappa: float | None = None, alpha: float | None = None, sigma: float | None = None):
        """
        Initialize the class

        :param kappa: mean reversion rate
        :param alpha: long term mean
        :param sigma: volatility coefficient
        """

        self.kappa = kappa
        self.alpha = alpha
        self.sigma = sigma

        self._validate_parameters()

        # test the feller condition
        if all(par is not None for par in self.parameters.values()):
            feller_condition = 2.0 * self.alpha * self.kappa >= self.sigma**2
            if not feller_condition:
                raise ValueError(
                    "The Feller condition (2*kappa*alpha>=sigma^2) is not "
                    "verified and with these params process could reach zero"
                )

    def _stationary_distribution(self, n_simulations: int = 1) -> np.ndarray:
        """
        Generate n_simulations values from the stationary distribution of a CIR process

        :params n_simulations: number of values simulated
        :return: the stationarity distribution
        """

        return gamma.rvs(
            2.0 * self.kappa * self.alpha / self.sigma**2, 0, self.sigma**2 / (2 * self.kappa), size=n_simulations
        )  # shape(must be >0), location, scale(must be >0)

    def _simulate(
        self, initial_value: float, n_steps: int, delta: float = 1.0, n_simulations: int = 1, method: str = "exact"
    ) -> pd.DataFrame:
        """
        Simulate CIR paths given params

        The time interval of simulation is [0,T] where T =n_steps*delta

        :param initial_value: starting point
        :param n_steps: sample size -> initial value + n observations
        :param delta: sampling interval
        :param n_simulations: number of paths simulated
        :param method: simulation method

        :return: n_simulations simulations of
            length n_steps + 1 (included the initial value)
        """

        if method not in ["exact", "euler"]:
            raise TypeError("not valid choice for `method`")
        # each column is a simulated path
        observations = initial_value * np.ones((n_steps + 1, n_simulations))

        ran = norm.rvs(size=(n_steps + 1, n_simulations))

        if method == "exact":
            # exact discretization
            kappa = np.ones(n_simulations) * self.kappa
            alpha = np.ones(n_simulations) * self.alpha
            sigma = np.ones(n_simulations) * self.sigma
            dof = np.array(4.0 * kappa * alpha / sigma**2)

            scale_factor = np.array((sigma**2 * (1.0 - np.exp(-kappa * delta))) / (4.0 * kappa))
            chi = np.random.chisquare(dof - 1, size=(n_steps + 1, n_simulations))

            dof = dof * np.ones(n_simulations)

            g = dof > 1  # condition different algorithm numerically advantageous
            b = dof <= 1  # bad algorithm

            if np.any(g):  # different algorithm numerically advantageous
                for i in range(n_steps):
                    item = observations[i, g] * np.exp(-kappa[g] * delta) / scale_factor[g]
                    observations[i + 1, g] = scale_factor[g] * ((ran[i + 1, g] + np.sqrt(item)) ** 2 + chi[i + 1, g])
            else:
                for i in range(n_steps):
                    item = observations[i, b] * np.exp(-kappa[b] * delta) / scale_factor[b]
                    N = np.random.poisson(item / 2, n_simulations)
                    chi = np.random.chisquare(dof[b] + 2 * N, n_simulations)
                    observations[i + 1, b] = scale_factor[b] * chi

        elif method == "euler":  # Euler scheme (full truncation)
            observations_h = initial_value * np.ones(
                (n_steps + 1, n_simulations)
            )  # needed for the euler approximation

            for i in range(n_steps):
                observations_h[i + 1, :] = (
                    observations_h[i, :]
                    + self.kappa * (self.alpha - np.maximum(0, observations_h[i, :])) * delta
                    + np.sqrt(np.maximum(0, observations_h[i, :])) * self.sigma * ran[i + 1, :] * np.sqrt(delta)
                )
                observations[i + 1] = np.maximum(0, observations_h[i + 1, :])

        else:
            raise TypeError("Not valid choice for `method`")

        return pd.DataFrame(observations)

    def _log_likelihood(self, observations, delta=1):
        """
        Log-likelihood function for CIR process

        • Vanyolos, Andras, Maxx Cho, and Scott Alan Glasgow.
            "Probability density of the CIR model."
            Available at SSRN 2508699 (2014).

        • Ahmed, Nafidi, and El Azri Abdenbi.
            "Inference in the stochastic Cox-Ingersol-Ross diffusion process
            with continuous sampling: Computational aspects and simulation."
            arXiv preprint arXiv:2103.15678 (2021).

        :param observations: columns indicates the different paths and rows indicates the observations
        :param delta: sampling interval
        :return: the maximum of a pseudo-log-likelihood function
        """

        if (observations.to_numpy() < 0).any():
            raise ValueError("The paths touches zero and it is impossible to calculate likelihood function")

        prices = observations.to_numpy().ravel()

        p, c = prices[:-1], prices[1:]
        e = np.exp(-self.kappa * delta)
        ss = self.sigma**2
        df = 4.0 * self.kappa * self.alpha / ss
        zeta = ss * (1.0 - e) / (4.0 * self.kappa)

        return ncx2.logpdf(c, df=df, nc=4.0 * self.kappa * e * p / (ss * (1.0 - e)), scale=zeta).sum()

    def _pseudo_maximum_likelihood_estimation(
        self, observations, delta: float = 1.0, to_array: bool = False
    ) -> dict | np.ndarray:
        """
        Compute the explicit expression for pseudo-maximum likelihood estimators proposed by Nowman(1997)

        :param observations: columns indicates the different paths and the rows indicates observations
        :param delta: sampling interval
        :param to_array: option to return an array instead of a dict

        :return: pseudo-mle parameters
        """

        n = len(observations) - 1

        observations = observations.to_numpy()

        obs_x = observations[0:n, :].sum(axis=0)[0]
        obs_x_1 = (observations[0:n, :] ** (-1)).sum(axis=0)[0]
        obs_y = observations[1 : n + 1, :].sum(axis=0)[0]
        obs_yyx_1 = ((observations[1 : n + 1, :] ** 2) * (observations[0:n] ** (-1))).sum(axis=0)[0]
        obs_x_1y = (observations[1 : n + 1, :] * (observations[0:n, :] ** (-1))).sum(axis=0)[0]

        # below the explicit expression for the parameters (kappa,alpha,sigma)
        n_inv = 1.0 / n
        b1 = (n_inv**2 * obs_y * obs_x_1 - n_inv * obs_x_1y) / (n_inv**2 * obs_x * obs_x_1 - 1.0)
        b1_minus = 1.0 - b1
        b2 = ((n_inv * obs_x_1y) - b1) / (b1_minus * n_inv * obs_x_1)
        b3 = n_inv * (
            obs_yyx_1
            + obs_x * (b1**2)
            + obs_x_1 * (b2**2) * (b1_minus**2)
            - 2 * obs_y * b1
            + 2 * n * b1 * b2 * b1_minus
            - 2 * obs_x_1y * b2 * b1_minus
        )

        # condition on b1 to get positive kappa, condition on b3 to get
        # sigma > 0 (delta is strictly positive)
        if 0.0 < b1 < 1.0 and b3 > 0.0:
            kappa = -(delta ** (-1)) * np.log(abs(b1))
            alpha = b2
            sigma = np.sqrt(2.0 * kappa * b3 / (1.0 - b1**2))

        else:
            kappa = np.nan
            alpha = np.nan
            sigma = np.nan

        if to_array:
            return np.array([kappa, alpha, sigma])
        return {"kappa": kappa, "alpha": alpha, "sigma": sigma}


if __name__ == "__main__":
    pass
