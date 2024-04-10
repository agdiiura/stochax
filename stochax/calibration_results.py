"""
----------------------
calibration_results.py
----------------------

A module for calibration results
"""

import logging

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

from numpy.random import RandomState
from sklearn.utils import Bunch
from plotly.subplots import make_subplots

__all__ = ["CalibrationResult"]
logger = logging.getLogger(__name__)


class CalibrationResult(object):
    """A class to store fit results and statistics"""

    _colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # middle gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    ]

    @property
    def process(self):
        """Return the process attribute"""
        return self._process

    @property
    def observations(self) -> pd.DataFrame:
        """Return the observations attribute"""
        return self._observations

    @property
    def method(self) -> str:
        """Return the method attribute"""
        return self._method

    def __init__(
        self,
        process: Any,
        observations: pd.DataFrame,
        delta: float = 1.0,
        method: str = "mle",
        n_boot_resamples: int = 1000,
        n_jobs: int = 2,
        seed: RandomState | int | None = None,
        bootstrap_results: pd.DataFrame | None = None,
    ):
        """
        Initialize the class

        :param process: a stochastic process instance
        :param observations: observations data
        :param delta: sampling interval
        :param method: choices are 'mle', 'parametric_bootstrap', 'non_parametric_bootstrap'
        :param n_boot_resamples: number bootstrap resamples
        :param n_jobs: number of parallel jobs
        :param seed: bootstrap random state
        :param bootstrap_results: a DataFrame contained the results of bootstrap procedure

        Example:

        .. code-block:: python

            ...
            res = process.calibrate(data)
            print(res.get_summary())

        """

        self._process = process

        if not self._process.is_calibrated:
            raise RuntimeError(f"The process is not calibrated: {self._process}")

        self._observations = observations
        self._delta = delta
        self._method = method
        self._n_boot_resamples = n_boot_resamples
        self._n_jobs = n_jobs
        self._seed = seed
        self._bootstrap_results = bootstrap_results

        self._msg = f"{self.__class__.__name__}({self._process}, observations.shape={self._observations.shape})"

    def __repr__(self) -> str:
        """Override the REPL output"""
        return self._msg

    def __str__(self) -> str:
        """Override the print output"""
        return self._msg

    def show_parameters(self) -> go.Figure:
        """
        Display parameters and relative errors

        :return fig: (go.Figure) obtained values
        """
        if "bootstrap" not in self._method:
            raise NotImplementedError("method not implemented with estimation `mle`")
        br = self._bootstrap_results.copy()

        keys = self.process.parameters.keys()
        fig = make_subplots(rows=len(keys), cols=1)

        idxs = [1 + v for v in range(len(keys))]
        charts = Bunch(**dict(zip(keys, idxs)))

        for (k, v), color in zip(self.process.parameters.items(), self._colors):
            x = br.loc[:, k].dropna().to_numpy()
            size = x.size
            if size > 0:
                n_bins = 2 * np.max([np.sqrt(size) + 1, np.log2(size) + 1, 25])
                bin_size = (x.max() - x.min()) / n_bins
                out = ff.create_distplot(
                    hist_data=[x], group_labels=[k], bin_size=bin_size, colors=[color]
                )
                for d in out.data:
                    fig.add_trace(d, row=charts[k], col=1)

            fig.add_vline(
                v,
                line_dash="dot",
                line_width=0.75,
                line_color="red",
                annotation_text=f"Estimated `{k}`: {v:.3}",
                annotation_position="top right",
                annotation_font_size=15,
                annotation_font_color="red",
                row=charts[k],
                col=1,
            )

        fig.update_layout(autosize=True, title_text="Estimated parameters")

        return fig

    def show_estimated_correlation(self) -> go.Figure:
        """
        Display the correlation obtained in the bootstrap procedure

        :return: the figure object
        """
        if "bootstrap" not in self._method:
            raise NotImplementedError("method not implemented with estimation `mle`")
        br = self._bootstrap_results.copy()

        fig = ff.create_scatterplotmatrix(br, diag="histogram", colormap="YlOrRd")

        fig.update_layout(autosize=True, title_text="Scatter-matrix bootstrap estimation")

        return fig

    def get_summary(self) -> dict:
        """
        Collect results and fit statistics.

        :return: Information about the fit.
        """

        summary = dict()

        ll = self.process.log_likelihood(self._observations)

        n_params = len(self.process.parameters.keys())
        n_obs = len(self._observations)

        summary["LogLikelihood"] = ll
        summary["AIC"] = 2.0 * n_params - 2.0 * ll
        summary["BIC"] = np.log(n_obs) * n_params - 2.0 * ll

        return summary


if __name__ == "__main__":
    pass
