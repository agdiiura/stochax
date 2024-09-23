"""
------------
test_core.py
------------

A test to check the core module

To run the code
$ python test_core.py
"""

import unittest

import numpy as np
import pandas as pd
import xmlrunner
import numpy.random

from test_stochax.config import xml_test_folder

from stochax.core import objective

rng = numpy.random.default_rng()


class MockProcess(object):
    """A class to represent a Stochastic-process"""

    def __init__(self, x: None | float = None):
        """Initialize the class"""
        self.x = x

    def log_likelihood(self, observations: pd.DataFrame, delta: float = 1) -> float:
        """Return the log-likelihood value"""
        return (self.x**2 - 1) / delta


class TestObjective(unittest.TestCase):
    """A class for objective function test"""

    def test_call(self):
        """Test the function call"""

        process = MockProcess

        n_runs = 10

        for _ in range(n_runs):
            value = objective(
                params=[rng.standard_normal()],
                process=process,
                observations=None,
                delta=0.5,
            )

            self.assertIsInstance(value, float)
            self.assertTrue(np.isfinite(value))


def build_suite() -> unittest.TestSuite:
    """Build the TestSuite"""
    suite = unittest.TestSuite()

    suite.addTest(TestObjective("test_call"))
    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
