# stochax 📈

Stochastic processes in python



[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

The file `pyproject.toml` contains the packages needed for the installation.
The code requires `python3.12+`.

### Installation
To install the package the simplest procedure is:
```bash
pip install stochax
```
Now you can test the installation... In a python shell:

```python
import stochax as sx

sx.__version__
```

#### Installation from source
Once you have cloned the repository
```bash
pip install .
```
To use the develop mode just write `pip install -e .`.


## Examples
### Data simulation
```python
import stochax as sx

abm = sx.ArithmeticBrownianMotion(mu=0.25, sigma=1.7)
realizations = abm.simulate(
    initial_value=1,
    n_steps=5
)

print(realizations)
```
```
          0
0  1.000000
1  3.567428
2  4.163523
3  4.874200
4  6.132376
5  5.651274
```
### Model fit
```python
import stochax as sx

abm = sx.ArithmeticBrownianMotion(mu=0.25, sigma=1.7)
realizations = abm.simulate(
    initial_value=1,
    n_steps=100
)
gbm = sx.GeometricBrownianMotion()
gbm.calibrate(realizations)

print(gbm)
```
```
GeometricBrownianMotion(mu=0.08278617288074738, sigma=0.33330614384487633)
```
Further examples can be found the [examples](examples) folder.
