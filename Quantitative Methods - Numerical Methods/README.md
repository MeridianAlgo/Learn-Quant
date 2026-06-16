# Quantitative Methods — Numerical Methods

Most of the formulas in finance cannot be solved with algebra. There is no
closed form for a bond's **yield-to-maturity** or an option's **implied
volatility** — you have a function and a target, and you must *search* for the
input that hits it. When a derivative or integral has no neat antiderivative,
you approximate it numerically instead.

This module implements the small toolkit that quietly powers the rest of the
repository — root finders, numerical differentiation and numerical integration —
written from first principles so you can see exactly how each black box behaves,
and where it can fail.

## Functions

| Function | Description |
|---|---|
| `bisection(f, a, b, tol, max_iter)` | Bracketed root finding — slow but cannot diverge |
| `newton_raphson(f, df, x0, tol, max_iter)` | Quadratic convergence using the derivative |
| `secant(f, x0, x1, tol, max_iter)` | Newton-like speed with no derivative needed |
| `finite_difference(f, x, h)` | Central-difference numerical derivative |
| `trapezoid(f, a, b, n)` | Composite trapezoid integration |
| `simpson(f, a, b, n)` | Composite Simpson's rule integration |

## Root finding — three trade-offs

| Method | Needs | Convergence | Robustness |
|---|---|---|---|
| Bisection | a sign-changing bracket | linear (~3.3 digits / 10 steps) | bulletproof |
| Newton-Raphson | a derivative + good guess | quadratic (digits double) | can diverge |
| Secant | two starting points | superlinear | can stall |

The practical rule: use **Newton** when you have a cheap derivative and a decent
guess, **secant** when you do not, and fall back to **bisection** when you need
a guarantee.

## Example

```python
import math
from numerical_methods import newton_raphson, secant, simpson, finite_difference

# Solve x^2 = 2 with and without a derivative.
f, df = lambda x: x*x - 2, lambda x: 2*x
print(newton_raphson(f, df, x0=1.0))     # 1.41421356...
print(secant(f, x0=1.0, x1=2.0))         # 1.41421356...

# Numerical derivative: d/dx sin(x) at x = 1 -> cos(1)
print(finite_difference(math.sin, 1.0))  # 0.5403...

# Integrate the standard normal density over [-3, 3] -> ~0.9973
pdf = lambda x: math.exp(-0.5*x*x) / math.sqrt(2*math.pi)
print(simpson(pdf, -3, 3))               # 0.99730...
```

## Differentiation by "bumping"

The central difference `(f(x+h) - f(x-h)) / 2h` has error of order `h²` — far
better than a one-sided difference. This is *exactly* how option **Greeks** are
often computed in practice: bump an input by a small `h`, re-price, and take the
difference. See `Finance - Greeks Calculator` for the finance-facing version.

## Integration — trapezoid vs. Simpson

The trapezoid rule joins points with straight lines (error `~1/n²`); Simpson's
rule fits parabolas (error `~1/n⁴`), so for smooth integrands like a probability
density Simpson reaches machine-level accuracy with far fewer points. Use
Simpson by default; reach for the trapezoid only when the integrand is jagged.

## Practical notes

- **Bracketing matters.** Bisection raises if `f(a)` and `f(b)` share a sign —
  that is a feature, not a bug: it refuses to lie about a root it cannot prove.
- **Newton can explode** near a flat spot (`f'(x) ≈ 0`). The implementation
  raises rather than returning nonsense.
- These routines are the engine behind `Finance - Implied Volatility Surface`
  (inverting Black-Scholes) and `Bond Price and Yield` (solving for YTM).
- For multi-dimensional problems, continue to
  [`Quantitative Methods - Optimization`](../Quantitative%20Methods%20-%20Optimization/).
