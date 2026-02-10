"""
Yield Curve Interpolation and Bootstrapping concepts.
"""

import numpy as np


def linear_interpolation(x, xp, fp):
    """
    Simulates linear interpolation for yields.
    """
    return np.interp(x, xp, fp)


if __name__ == "__main__":
    # Known market points (Maturity in years, Yield in %)
    tenors = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    yields = [0.051, 0.048, 0.045, 0.042, 0.041, 0.043]

    # Points we want to interpolate
    target_tenors = [0.5, 3.0, 7.0, 20.0]

    print("Yield Curve Interpolation (Linear)")
    print("-" * 40)
    print(f"{'Maturity':<10} | {'Yield (%)':<10}")
    print("-" * 40)

    # Print known points
    for t, y in zip(tenors, yields):
        print(f"{t:10.2f} | {y:10.2%}")

    print("-" * 40)
    print("Interpolated Points:")
    for t in target_tenors:
        y_interp = linear_interpolation(t, tenors, yields)
        print(f"{t:10.2f} | {y_interp:10.2%} (Estimated)")
