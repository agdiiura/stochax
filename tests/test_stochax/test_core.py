"""
----------------------------
test_stochastic_processes.py
----------------------------

A test to check the test_stochx module

To run the code
$ python test_stochastic_processes.py
"""

import unittest

import numpy as np
import xmlrunner

from test_stochax.config import xml_test_folder

from stochax.core import Bounds, ParameterBound

VERBOSE = True
rng = np.random.default_rng()


class TestParameterBound(unittest.TestCase):
    """Test the Bound class"""

    def test_call(self):
        """Test the call method"""

        with self.assertRaises(ValueError):
            ParameterBound("err", float, 100, -100)

        b = ParameterBound("a", float, -1, 1)
        with self.assertRaises(TypeError):
            b(100)

        with self.assertRaises(ValueError):
            b(100.0)

        with self.assertRaises(ValueError):
            b(-100.0)

        self.assertIsNone(b(None))

    def test_eq(self):
        """Test the __eq__ method"""

        a = ParameterBound("a", float, -1, 7)
        b = ParameterBound("a", float, -1, 7)

        self.assertEqual(a, b)

        self.assertFalse(a == "string")

    def test_ne(self):
        """Test the __ne__ method"""

        a = ParameterBound("a", float, -1, 7)
        b = ParameterBound("a", float, -1, 100)

        self.assertNotEqual(a, b)


class TestBounds(unittest.TestCase):
    """Test the Bounds class"""

    def test_call(self):
        """Test the call method"""

        with self.assertRaises(ValueError):
            Bounds(ParameterBound("a", float, -1, 1), ParameterBound("a", float, 0, 1))

        b = Bounds(ParameterBound("a", float, -1, 1), ParameterBound("b", float, 0, 1))

        with self.assertRaises(ValueError):
            b({"a": 100.0, "b": -100.0})

        with self.assertRaises(KeyError):
            b({"c": 1.0})

        self.assertIsNone(b({"a": 0.0, "b": 0.1}))

    def test_len(self):
        """Test the __len__ method"""

        vals = [ParameterBound("a", -1, 1), ParameterBound("b", 0, 1)]
        b = Bounds(*vals)

        self.assertEqual(len(b), len(vals))

    def test_to_tuple(self):
        """Test the to_tuple method"""

        b = Bounds(ParameterBound("a", float, -1, 1), ParameterBound("b", float, 0, 1))
        t = b.to_tuple()

        self.assertEqual(t, [(-1, 1), (0, 1)])


def build_suite():
    """Build the TestSuite"""
    suite = unittest.TestSuite()

    suite.addTest(TestParameterBound("test_call"))
    suite.addTest(TestParameterBound("test_eq"))
    suite.addTest(TestParameterBound("test_ne"))

    suite.addTest(TestBounds("test_call"))
    suite.addTest(TestBounds("test_len"))
    suite.addTest(TestBounds("test_to_tuple"))

    return suite


if __name__ == "__main__":
    """The main script"""

    runner = xmlrunner.XMLTestRunner(output=xml_test_folder)
    runner.run(build_suite())
