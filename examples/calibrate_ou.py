"""
---------------
calibrate_ou.py
---------------

Calibrate an Ornstein-Uhlenbeck model on BOT6M
"""

import pandas as pd

from stochax import OrnsteinUhlenbeck

if __name__ == "__main__":
    data = pd.read_csv("data/bot6m.csv", index_col="Timestamp", parse_dates=True)
    print(f"Show dataset for BOT@6M\n{data}")

    delta = 1

    ou = OrnsteinUhlenbeck()

    res = ou.calibrate(data, delta=delta, method="mle")

    print(f"\n\n> Summary MLE:\n{ou.parameters}\n{res.get_summary()}")

    res = ou.calibrate(data, delta=delta, method="parametric_bootstrap")

    print(f"\n\n> Summary Parametric-Bootstrap:\n{ou.parameters}\n{res.get_summary()}")

    print("\n\nDone!")
