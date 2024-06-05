***********
stochax
***********

**Last Update**: |today|

**Version**: |version|

:mod: A package for calibration and simulation of stochastic processes in python.

Installation
~~~~~~~~~~~~~~~~

Run the install command

.. code-block:: bash
   :linenos:

   pip install stochax

.. currentmodule:: stochax

Example
~~~~~~~~~~~
Simulate a simple stochastic process

.. code-block:: python
   :linenos:

     from stochax import ArithmeticBrownianMotion

     abm = ArithmeticBrownianMotion(mu=0, sigma=0.5)
     paths = abm.simulate(
         initial_value=0.5,
         n_steps=52,
         delta=1/52,
         n_simulations=100
     )

Load data from an external file and calibrate
a Geometric Brownian Motion

.. code-block:: python
   :linenos:

     import pandas as pd
     from stochax import GeometricBrownianMotion

     data = pd.read_csv('path/to/data.csv')
     gbm = GeometricBrownianMotion()
     res = gbm.calibrate(data)

     print(res.get_summary())



stochax
~~~~~~~~~~~
.. autosummary::
   :toctree: stochax/

   stochax.core
   stochax.calibration_results
   stochax.brownian_motion
   stochax.mean_reverting
