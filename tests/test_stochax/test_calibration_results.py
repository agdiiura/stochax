"""
--------------------------
test_calibration_result.py
--------------------------

A test to check the calibration_result module

To run the code
$ python test_calibration_result.py
"""

import unittest

import pandas as pd
import xmlrunner
import plotly.graph_objs as go

from test_stochax.config import xml_test_folder, simulate_univariate_process

from stochax.calibration_results import CalibrationResult


class MockProcess(object):
    """A class to represent a Stochastic-process"""

    @property
    def parameters(self) -> dict:
        """Return the parameters attribute"""
        return dict(a=self.a, b=self.b)

    @property
    def is_calibrated(self) -> bool:
        """Return the is_calibrated flag"""
        return all(v is not None for v in self.parameters.values())

    def __init__(self, a=None, b=None):
        """Initialize the class"""

        self.a = a
        self.b = b

    def log_likelihood(self, observations: pd.DataFrame) -> float:
        """Return a fake value"""
        return 42.42


class TestCalibrationResult(unittest.TestCase):
    """A class to test the CalibrationResult object"""

    observations = pd.DataFrame({"obs": simulate_univariate_process(250)})

    @classmethod
    def setUpClass(cls):
        """Configure the test"""

        process = MockProcess(a=1.0, b=2.0)
        cls.delta = 1
        cls.method = "fake_bootstrap"
        cls.bootstrap_results = pd.DataFrame(
            {k: simulate_univariate_process(100) for k in process.parameters.keys()}
        )

        cls.fit = CalibrationResult(
            process=process,
            observations=cls.observations,
            delta=cls.delta,
            method=cls.method,
            bootstrap_results=cls.bootstrap_results,
        )

    def _assert_show(self, method: str):
        """Assert the goodness of show methods"""

        m = getattr(self.fit, f"show_{method}")
        obj = m()

        self.assertIsInstance(obj, go.Figure, msg=f"Error with method `show_{method}`")

    def test_init(self):
        """Test the class construction"""

        obs = self.fit.observations
        self.assertIsNone(pd.testing.assert_frame_equal(self.observations, obs))

        m = self.fit.method
        self.assertEqual(m, self.method)

        self.assertIsInstance(str(self.fit), str)

        wrong_process = MockProcess()

        with self.assertRaises(RuntimeError):
            CalibrationResult(
                process=wrong_process,
                observations=self.observations,
                delta=self.delta,
                method=self.method,
                bootstrap_results=self.bootstrap_results,
            )

    def test_show_parameters(self):
        """Test the show_parameters method"""
        self._assert_show(method="parameters")

    def test_show_estimated_correlation(self):
        """Test the show_estimated_correlation method"""
        self._assert_show(method="estimated_correlation")

    def test_get_summary(self):
        """Test the get_summary method"""

        s = self.fit.get_summary()
        self.assertIsInstance(s, dict)

        expected = {
            "LogLikelihood",
            "n_parameters",
            "n_observations",
            "AIC",
            "BIC",
            "HQC",
        }
        self.assertEqual(expected, set(s.keys()))


def build_suite() -> unittest.TestSuite:
    """Build the TestSuite"""
    suite = unittest.TestSuite()

    suite.addTest(TestCalibrationResult("test_init"))
    suite.addTest(TestCalibrationResult("test_show_parameters"))
    suite.addTest(TestCalibrationResult("test_show_estimated_correlation"))
    suite.addTest(TestCalibrationResult("test_get_summary"))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
