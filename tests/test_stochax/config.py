"""
---------
config.py
---------

The configuration file for tests
"""

import unittest

from pathlib import Path

import numpy as np
import xmlrunner

from numpy.random import default_rng

__all__ = ["xml_test_folder", "simulate_univariate_process"]

config_path = Path(__file__).absolute().parent


def simulate_univariate_process(
    size: int,
    baseline: float = 100.0,
    std: float = 1.0,
    min_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Create an array of simulated prices from Student-T returns with high ddf
    to be gaussian-like.

    Args:
        size: length of prices
        baseline: starting value of the price
        std: standard deviation scale
        min_value: minimum allowed price value
        rng: numpy Generator

    Returns:
        prices array

    """

    if rng is None:
        rng = default_rng()

    returns = rng.standard_t(8, size=size)
    return np.maximum(baseline + std * returns.cumsum(), min_value)


def get_xml_test_folder() -> str:
    """
    Serve a test-reports folder based on the current system environment

    :return: xml_test_folder: default test report folder
    """
    path = config_path.parent / "test-reports"
    path.mkdir(exist_ok=True)

    return str(path)


xml_test_folder = get_xml_test_folder()


class TestConfig(unittest.TestCase):
    """The test class for configuration file"""

    def test_all(self):
        """Assert that test are allowed"""

        for name in __all__:
            self.assertIn(name, globals())

    def test_simulate_univariate_process(self):
        """Test the simulate_univariate_process function"""

        n_tests = 5
        rng = default_rng()

        for k in range(n_tests):
            size = rng.integers(low=10, high=500)
            std = rng.exponential()
            min_value = rng.standard_normal()
            baseline = rng.exponential(scale=100)

            prices = simulate_univariate_process(
                size, std=std, min_value=min_value, baseline=baseline
            )
            self.assertIsInstance(prices, np.ndarray)
            self.assertEqual(prices.shape, (size,))
            self.assertTrue((prices >= min_value).all())


def build_suite():
    """Construct the TestSuite"""

    suite = unittest.TestSuite()

    suite.addTest(TestConfig("test_all"))
    suite.addTest(TestConfig("test_simulate_univariate_process"))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
