# stochax 📈

A python package for the simulation and calibration of
stochastic processes.

## Minimal examples

### Simulation

It can be used to simulate a stochastic
process:

```python

from stochax import ArithmeticBrownianMotion
abm = ArithmeticBrownianMotion(mu=0, sigma=0.5)
paths = abm.simulate(
    initial_value=0.5,
    n_steps=52,
    delta=1/52,
    n_simulations=100
)
```

### Model fit

It is also possible to use the package to
fit data:

```python
import pandas as pd
from stochax import GeometricBrownianMotion

data = pd.read_csv('path/to/data.csv')
gbm = GeometricBrownianMotion()
res = gbm.calibrate(data)

print(res.get_summary())
```

## Installation

To install the package the simplest procedure is:
```bash
pip install stochax
```
Now you can test the installation... In a python shell:

```python
import stochax as sx

sx.__version__
```
