"""
----------------------------
test_stochastic_processes.py
----------------------------

A test to check the stochastic process contained in stochax

To run the code
$ python test_stochastic_processes.py
"""

import os
import unittest

import numpy as np
import pandas as pd
import xmlrunner

from test_stochax.config import xml_test_folder, simulate_univariate_process

from stochax.mean_reverting import CoxIngersollRoss, OrnsteinUhlenbeck
from stochax.brownian_motion import (GeometricBrownianMotion,
                                     ArithmeticBrownianMotion)
from stochax.calibration_results import CalibrationResult

VERBOSE = os.environ.get("VERBOSE", True)
rng = np.random.default_rng()


class TestStochasticProcess(unittest.TestCase):
    """The base class for StochasticProcess"""

    process = None
    init_kwargs = None
    simulate_kwargs = None
    calibrate_kwargs = None

    # number of simulations
    n_simulations = 10
    n_simulations_large = 10000
    # number of steps
    n_steps = 5
    n_steps_large = 5000
    # initial value
    initial_value = 1
    # observations, as a simple GBM with Student-T noise
    observations = pd.DataFrame({"obs": simulate_univariate_process(250)})

    def test_init(self):
        """Test the initializer"""
        process = self.process()
        params = process.parameters
        self.assertTrue(all(x is None for x in params.values()))

        process = self.process(**self.init_kwargs)
        params = process.parameters
        for key, val in self.init_kwargs.items():
            self.assertEqual(val, params[key], msg=f"Error with `{key}`")

    def test_stationary_distribution(self):
        """Test the stationary_distribution method, if the class is provided"""

        process = self.process()

        if hasattr(process, "stationary_distribution"):
            with self.assertRaises(TypeError):
                process.stationary_distribution()

            process = self.process(**self.init_kwargs)
            dist = process.stationary_distribution(n_simulations=self.n_simulations)
            self.assertIsInstance(dist, np.ndarray)
            self.assertTrue(len(dist), self.n_simulations)

    def test_simulate(self):
        """Test the simulate method"""

        process = self.process()
        with self.assertRaises(TypeError):
            process.simulate(initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps)

        process = self.process(**self.init_kwargs)

        with self.assertRaises(ValueError):
            process.simulate(
                initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps, delta=-1
            )

        with self.assertRaises(ValueError):
            process.simulate(initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=-1)

        with self.assertRaises(ValueError):
            process.simulate(initial_value=self.initial_value, n_simulations=-1, n_steps=self.n_steps)

        kw = self.init_kwargs.copy()
        key = list(kw.keys())[-1]
        with self.assertRaises(TypeError):
            kw[key] = None
            process = self.process(**kw)
            process.simulate(initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps)

        with self.assertRaises(TypeError):
            kw[key] = np.nan
            process = self.process(**kw)
            process.simulate(initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps)

        with self.assertRaises(TypeError):
            kw[key] = np.inf
            process = self.process(**kw)
            process.simulate(initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps)

        for kw in self.simulate_kwargs:
            process = self.process(**self.init_kwargs)
            simulations = process.simulate(
                initial_value=self.initial_value, n_simulations=self.n_simulations, n_steps=self.n_steps, **kw
            )

            self.assertIsInstance(simulations, pd.DataFrame)
            self.assertFalse(simulations.empty)
            self.assertEqual(simulations.shape, (self.n_steps + 1, self.n_simulations))
            self.assertEqual(simulations.iloc[0].tolist(), self.n_simulations * [self.initial_value])
            self.assertFalse(simulations.isna().any().any())

            simulations = process.simulate(
                initial_value=self.initial_value, n_simulations=self.n_simulations_large, n_steps=self.n_steps_large
            )

            # if the process has a stationary distribution test against it
            if hasattr(process, "stationary_distribution"):
                stationary = process.stationary_distribution(n_simulations=self.n_simulations_large)
                stationary_mean, stationary_std = np.mean(stationary), np.std(stationary)
                last_simulations_mean = simulations.iloc[-1].mean()
                self.assertTrue(
                    stationary_mean - stationary_std < last_simulations_mean < stationary_mean + stationary_std
                )

    def test_calibrate(self):
        """Test the calibrate method"""

        process = self.process()

        with self.assertRaises(TypeError):
            process.calibrate("string")

        with self.assertRaises(ValueError):
            process.calibrate(pd.DataFrame())

        with self.assertRaises(ValueError):
            process.calibrate([1, 1, 1, 2, np.nan])

        with self.assertRaises(TypeError):
            process.calibrate(self.observations, method="fake")

        with self.assertRaises(ValueError):
            process.calibrate(self.observations, delta=-1)

        with self.assertRaises(ValueError):
            process.calibrate(self.observations, n_boot_resamples=-1)

        with self.assertRaises(ValueError):
            process.calibrate(self.observations.head(1))

        for kw in self.calibrate_kwargs:
            r = process.calibrate(observations=self.observations, **kw)
            for key, val in process.parameters.items():
                self.assertIsInstance(val, float, msg=f'Error in {kw["method"]}')
                self.assertTrue(pd.notnull(val))

            self.assertIsInstance(r, CalibrationResult)

            if VERBOSE:
                print(
                    f'\n#############################'
                    f'\nUsing method `{kw["method"]}`\n'
                    f'Estimated process:\n{process}'
                )
                print("\n".join(f" - {k}: {v:.8}" for k, v in r.get_summary().items()))

    def test_log_likelihood(self):
        """Test the log_likelihood method"""
        process = self.process()
        with self.assertRaises(TypeError):
            process.log_likelihood(observations=self.observations)

        process = self.process(**self.init_kwargs)
        ll = process.log_likelihood(observations=self.observations)
        self.assertIsInstance(ll, float)

    def test_copy(self):
        """Test the copy method"""
        process = self.process(**self.init_kwargs)
        obj = process.copy()

        self.assertIsInstance(obj, process.__class__)
        self.assertIsNot(obj, process)

        for key, val in process.parameters.items():
            att = getattr(obj, key)
            self.assertEqual(att, val)


class TestOrnsteinUhlenbeck(TestStochasticProcess):
    """The class for OrnsteinUhlenbeck test"""

    process = OrnsteinUhlenbeck
    init_kwargs = {
        "kappa": rng.normal(loc=1.2, scale=0.01),
        "alpha": rng.normal(loc=0.9, scale=0.01),
        "sigma": rng.normal(loc=0.9, scale=0.01),
    }
    calibrate_kwargs = [
        dict(method="mle"),
        dict(method="parametric_bootstrap"),
        dict(method="non_parametric_bootstrap"),
        dict(method="numerical_mle"),
    ]
    simulate_kwargs = [dict()]


class TestCoxIngersollRoss(TestStochasticProcess):
    """The class for CoxIngersollRoss test"""

    process = CoxIngersollRoss
    init_kwargs = {
        "kappa": rng.normal(loc=1.2, scale=0.01),
        "alpha": rng.normal(loc=0.9, scale=0.01),
        "sigma": rng.normal(loc=0.9, scale=0.01),
    }
    calibrate_kwargs = [
        dict(method="pseudo_mle"),
        dict(method="parametric_bootstrap"),
        dict(method="non_parametric_bootstrap"),
        dict(method="numerical_mle"),
    ]
    simulate_kwargs = [dict(method="exact"), dict(method="euler")]


class TestArithmeticBrownianMotion(TestStochasticProcess):
    """The class for ArithmeticBrownianMotion test"""

    process = ArithmeticBrownianMotion
    init_kwargs = {"mu": rng.normal(loc=0, scale=0.01), "sigma": rng.normal(loc=0.9, scale=0.01)}
    calibrate_kwargs = [
        dict(method="mle"),
        dict(method="parametric_bootstrap"),
        dict(method="non_parametric_bootstrap"),
        dict(method="numerical_mle"),
    ]
    simulate_kwargs = [dict()]


class TestGeometricBrownianMotion(TestStochasticProcess):
    """The class for GeometricBrownianMotion test"""

    process = GeometricBrownianMotion
    init_kwargs = {"mu": rng.normal(loc=0, scale=0.01), "sigma": rng.normal(loc=0.9, scale=0.01)}
    calibrate_kwargs = [
        dict(method="mle"),
        dict(method="parametric_bootstrap"),
        dict(method="non_parametric_bootstrap"),
        dict(method="numerical_mle"),
    ]
    simulate_kwargs = [dict()]


def build_suite(model: str = "all"):
    """Build the TestSuite"""
    suite = unittest.TestSuite()

    tests = [
        "test_init",
        "test_stationary_distribution",
        "test_simulate",
        "test_calibrate",
        "test_log_likelihood",
        "test_copy",
    ]

    models = {
        "ou": TestOrnsteinUhlenbeck,
        "cir": TestCoxIngersollRoss,
        "abm": TestArithmeticBrownianMotion,
        "gbm": TestGeometricBrownianMotion,
    }
    classes = models.values() if model == "all" else [models[model]]

    for cls in classes:
        for test in tests:
            suite.addTest(cls(test))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
