"""
-----------------
simulation_gbm.py
-----------------

Simulate a GBM and then fit the parameters
"""

import stochax as sx

if __name__ == "__main__":
    gbm = sx.GeometricBrownianMotion(mu=1.0, sigma=1.5)
    dt = 1 / 365

    data = gbm.simulate(initial_value=1, n_steps=365, delta=dt)

    print(f"Simulated data using {gbm}:\n{data}")

    # calibrate the data
    res = gbm.calibrate(data, method="mle", delta=dt)

    print(f"Estimated parameters: {res.process.parameters}")
    print("\nDone!")
