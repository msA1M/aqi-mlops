import pandas as pd
import numpy as np

def calculate_psi(expected, actual, bins=10):
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()

    if expected.empty or actual.empty:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents)
        * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )
    return psi
