# 📋 API Reference - Complete Function and Class Documentation

## 🎯 Comprehensive Reference Guide

This reference guide provides detailed documentation for all utilities, functions, classes, and methods in the Quantitative Finance Learning Platform.

---

## 📚 Utility Reference

### **Data Structures**

#### **Arrays (NumPy Operations)**
**Location**: `UTILS - Data Structures - Arrays/arrays.py`

**Class: ArrayOperations**
```python
class ArrayOperations:
    """Comprehensive array operations for quantitative finance."""

    def __init__(self):
        """Initialize with random number generator."""

    def create_price_series(self, start_price: float, periods: int,
                          drift: float = 0.0, volatility: float = 0.1) -> np.ndarray:
        """Create simulated price series using geometric Brownian motion."""

    def calculate_returns(self, prices: np.ndarray,
                         method: str = 'arithmetic') -> np.ndarray:
        """Calculate returns from price series."""

    def portfolio_return(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Calculate portfolio return given weights and asset returns."""

    def portfolio_volatility(self, weights: np.ndarray,
                           cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility."""

    def covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix from returns data."""

    def correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix from returns data."""

    def optimal_portfolio_weights(self, expected_returns: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 risk_free_rate: float = 0.02) -> np.ndarray:
        """Calculate optimal portfolio weights using mean-variance optimization."""

    def monte_carlo_simulation(self, s0: float, mu: float, sigma: float,
                              t: float, n_paths: int, n_steps: int) -> np.ndarray:
        """Perform Monte Carlo simulation for asset price paths."""

    def value_at_risk(self, returns: np.ndarray,
                     confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation."""

    def expected_shortfall(self, returns: np.ndarray,
                          confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall using historical simulation."""

    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Perform linear regression analysis."""

    def matrix_operations_demo(self) -> None:
        """Demonstrate matrix operations useful in finance."""

    def statistical_analysis(self, data: np.ndarray) -> dict:
        """Perform comprehensive statistical analysis."""

    def hypothesis_testing(self, sample1: np.ndarray, sample2: np.ndarray,
                          test_type: str = 't_test') -> dict:
        """Perform hypothesis testing between two samples."""
```

#### **Lists (Python List Operations)**
**Location**: `UTILS - Data Structures - Lists/lists.py`

**Class: ListOperations**
```python
class ListOperations:
    """Comprehensive list operations for financial applications."""

    def __init__(self):
        """Initialize with empty transaction log."""

    def create_portfolio_list(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create structured portfolio list from asset data."""

    def calculate_portfolio_weights(self, portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate portfolio weights based on current values."""

    def add_transaction(self, tx_type: str, ticker: str, shares: int,
                       price: float, timestamp: Optional[str] = None) -> None:
        """Add transaction to transaction log."""

    def get_transactions_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """Get all transactions for specific ticker."""

    def calculate_position_pnl(self, ticker: str, current_price: float) -> Dict[str, Any]:
        """Calculate profit/loss for a position."""

    def sort_portfolio_by_metric(self, portfolio: List[Dict[str, Any]],
                                metric: str, reverse: bool = True) -> List[Dict[str, Any]]:
        """Sort portfolio by specific metric."""

    def filter_portfolio(self, portfolio: List[Dict[str, Any]],
                        min_value: float = 0, max_value: float = float('inf'),
                        sector: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter portfolio based on criteria."""

    def calculate_portfolio_statistics(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio statistics."""

    def moving_average_strategy(self, prices: List[float],
                               short_window: int = 5, long_window: int = 20) -> List[str]:
        """Simple moving average crossover strategy."""

    def rebalance_portfolio(self, current_portfolio: List[Dict[str, Any]],
                           target_weights: Dict[str, float],
                           current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Rebalance portfolio to target weights."""

    def performance_analysis(self, portfolio_values: List[float]) -> Dict[str, Any]:
        """Analyze portfolio performance metrics."""

    def list_comprehension_examples(self) -> None:
        """Demonstrate advanced list comprehension techniques."""
```

#### **Dictionaries (Key-Value Data Structures)**
**Location**: `UTILS - Data Structures - Dictionaries/dictionaries.py`

**Class: DictionaryOperations**
```python
class DictionaryOperations:
    """Comprehensive dictionary operations for financial applications."""

    def __init__(self):
        """Initialize with empty databases."""

    def create_asset_database(self, assets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create structured asset database from asset data."""

    def update_asset_price(self, ticker: str, price: float,
                          volume: Optional[int] = None) -> None:
        """Update asset price and related data."""

    def get_asset_info(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive asset information."""

    def search_assets(self, criteria: Dict[str, Any]) -> List[str]:
        """Search assets based on criteria."""

    def create_portfolio_registry(self, portfolios: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create portfolio registry for multiple portfolios."""

    def calculate_portfolio_value(self, portfolio_id: str) -> float:
        """Calculate current portfolio value."""

    def get_portfolio_allocation(self, portfolio_id: str) -> Dict[str, float]:
        """Get portfolio allocation by asset."""

    def create_market_data_cache(self, market_data: Dict[str, Any]) -> None:
        """Create market data cache for performance."""

    def get_market_data(self, ticker: str, data_type: str = 'price') -> Any:
        """Get cached market data."""

    def dictionary_comprehension_examples(self) -> None:
        """Demonstrate dictionary comprehension techniques."""

    def merge_financial_data(self, *data_dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple financial data dictionaries."""

    def export_to_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Export dictionary data to JSON file."""

    def import_from_json(self, filename: str) -> Dict[str, Any]:
        """Import dictionary data from JSON file."""

    def performance_comparison(self) -> None:
        """Compare performance of different dictionary operations."""
```

### **Quantitative Methods**

#### **Time Value of Money (TVM)**
**Location**: `UTILS - Quantitative Methods - TVM/tvm_calculator.py`

**Class: TimeValueMoney**
```python
class TimeValueMoney:
    """Comprehensive Time Value of Money calculations."""

    def __init__(self):
        """Initialize the TimeValueMoney class."""

    def future_value_single(self, pv: float, rate: float, periods: int,
                          compounding: int = 1) -> float:
        """Calculate future value of single present value."""

    def present_value_single(self, fv: float, rate: float, periods: int,
                           compounding: int = 1) -> float:
        """Calculate present value of single future value."""

    def future_value_annuity(self, pmt: float, rate: float, periods: int,
                           type: int = 0, compounding: int = 1) -> float:
        """Calculate future value of ordinary annuity."""

    def present_value_annuity(self, pmt: float, rate: float, periods: int,
                            type: int = 0, compounding: int = 1) -> float:
        """Calculate present value of ordinary annuity."""

    def annuity_payment(self, pv: float, rate: float, periods: int,
                       type: int = 0, compounding: int = 1) -> float:
        """Calculate required payment for loan or annuity."""

    def net_present_value(self, initial_investment: float,
                         cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value of investment."""

    def internal_rate_of_return(self, initial_investment: float,
                              cash_flows: List[float], guess: float = 0.1) -> Optional[float]:
        """Calculate Internal Rate of Return."""

    def profitability_index(self, initial_investment: float,
                          cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Profitability Index."""

    def loan_payment(self, principal: float, annual_rate: float,
                    years: int, payments_per_year: int = 12) -> float:
        """Calculate loan payment amount."""

    def loan_amortization_schedule(self, principal: float, annual_rate: float,
                                  years: int, payments_per_year: int = 12) -> List[Dict[str, float]]:
        """Generate loan amortization schedule."""

    def effective_annual_rate(self, nominal_rate: float, compounding_freq: int) -> float:
        """Calculate effective annual rate."""

    def continuous_compounding(self, pv: float, rate: float, time: float) -> float:
        """Calculate future value with continuous compounding."""

    def bond_price(self, face_value: float, coupon_rate: float, years: int,
                  market_rate: float, payments_per_year: int = 2) -> float:
        """Calculate bond price."""

    def bond_yield_to_maturity(self, bond_price: float, face_value: float,
                              coupon_rate: float, years: int,
                              payments_per_year: int = 2, guess: float = 0.05) -> Optional[float]:
        """Calculate bond yield to maturity."""

    def retirement_planning(self, current_age: int, retirement_age: int,
                          current_savings: float, annual_contribution: float,
                          expected_return: float, desired_income: float,
                          life_expectancy: int, inflation_rate: float = 0.03) -> Dict[str, float]:
        """Comprehensive retirement planning analysis."""

    def sensitivity_analysis(self, base_scenario: Dict[str, float],
                           variable: str, values: List[float]) -> List[Dict[str, float]]:
        """Perform sensitivity analysis on TVM calculations."""

    def create_comparison_table(self, scenarios: List[Dict[str, str]]) -> str:
        """Create formatted comparison table for different scenarios."""
```

#### **Statistics**
**Location**: `UTILS - Quantitative Methods - Statistics/statistical_analysis.py`

**Class: StatisticalAnalysis**
```python
class StatisticalAnalysis:
    """Comprehensive statistical analysis for quantitative finance."""

    def __init__(self):
        """Initialize with random number generator."""

    def calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics."""

    def calculate_financial_returns(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate financial return statistics."""

    def calculate_value_at_risk(self, returns: np.ndarray,
                               confidence_level: float = 0.95,
                               method: str = 'historical') -> float:
        """Calculate Value at Risk using different methods."""

    def calculate_expected_shortfall(self, returns: np.ndarray,
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall."""

    def calculate_max_drawdown(self, price_series: np.ndarray) -> float:
        """Calculate maximum drawdown from price series."""

    def normality_tests(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform normality tests on data."""

    def correlation_analysis(self, returns_matrix: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """Analyze correlations between multiple assets."""

    def hypothesis_testing(self, sample1: np.ndarray, sample2: np.ndarray,
                          test_type: str = 'parametric') -> Dict[str, Dict[str, float]]:
        """Perform hypothesis tests between two samples."""

    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform linear regression analysis."""

    def monte_carlo_simulation(self, initial_value: float, expected_return: float,
                              volatility: float, time_horizon: float,
                              n_simulations: int = 10000) -> Dict[str, Union[float, np.ndarray]]:
        """Perform Monte Carlo simulation for portfolio valuation."""

    def distribution_analysis(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Analyze various probability distributions."""

    def rolling_statistics(self, data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """Calculate rolling statistics for time series."""

    def portfolio_performance_metrics(self, portfolio_returns: np.ndarray,
                                    benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""

    def generate_sample_data(self, distribution: str = 'normal',
                           n_samples: int = 1000, **params) -> np.ndarray:
        """Generate sample data from various distributions."""
```

---

## 📊 Mathematical Formulas

### **Time Value of Money**
- **Future Value (Single Sum)**: `FV = PV × (1 + r)^n`
- **Present Value (Single Sum)**: `PV = FV / (1 + r)^n`
- **Future Value (Annuity)**: `FV = PMT × [(1 + r)^n - 1] / r`
- **Present Value (Annuity)**: `PV = PMT × [1 - 1/(1 + r)^n] / r`
- **Loan Payment**: `PMT = PV × [r(1 + r)^n] / [(1 + r)^n - 1]`

### **Statistical Measures**
- **Arithmetic Mean**: `μ = Σx_i / n`
- **Standard Deviation**: `σ = √[Σ(x_i - μ)² / (n-1)]`
- **Sharpe Ratio**: `SR = (R_p - R_f) / σ_p`
- **Value at Risk**: `VaR = -μ - σ × z_α`
- **Beta**: `β = Cov(R_i, R_m) / Var(R_m)`

### **Portfolio Theory**
- **Portfolio Return**: `R_p = Σ(w_i × R_i)`
- **Portfolio Variance**: `σ_p² = ΣΣ(w_i × w_j × σ_i × σ_j × ρ_ij)`
- **CAPM**: `R_i = R_f + β_i × (R_m - R_f)`

### **Options Pricing**
- **Black-Scholes**: `C = S₀N(d₁) - Ke^(-rt)N(d₂)`
- **d₁**: `(ln(S₀/K) + (r + σ²/2)t) / (σ√t)`
- **d₂**: `d₁ - σ√t`

---

## 🔧 Error Codes and Exceptions

### **Custom Exceptions**
```python
class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds."""

class InvalidTickerError(Exception):
    """Raised when ticker symbol is invalid."""

class MarketDataError(Exception):
    """Raised when market data is unavailable."""

class CalculationError(Exception):
    """Raised when financial calculation fails."""
```

### **Error Handling Patterns**
```python
def safe_division(dividend, divisor):
    """Safely divide two numbers with error handling."""
    try:
        result = dividend / divisor
        return result
    except ZeroDivisionError:
        raise CalculationError("Cannot divide by zero")
    except TypeError:
        raise TypeError("Invalid data types for division")

def validate_ticker(ticker):
    """Validate stock ticker with comprehensive checks."""
    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string")
    if not ticker.isalpha():
        raise InvalidTickerError("Ticker must contain only letters")
    if len(ticker) > 5 or len(ticker) < 1:
        raise InvalidTickerError("Ticker must be 1-5 characters long")
    return ticker.upper()
```

---

## 📈 Performance Benchmarks

### **Algorithm Complexity**
| Operation | Data Structure | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|------------------|
| Access | List | O(1) | O(1) |
| Search | List | O(n) | O(1) |
| Insert | List | O(n) | O(1) |
| Delete | List | O(n) | O(1) |
| Access | Dictionary | O(1) | O(1) |
| Search | Dictionary | O(1) | O(1) |
| Insert | Dictionary | O(1) | O(1) |
| Delete | Dictionary | O(1) | O(1) |

### **NumPy Performance**
- **Vectorized Operations**: 10-100x faster than loops
- **Memory Layout**: C-contiguous arrays for optimal performance
- **Broadcasting**: Automatic dimension alignment
- **BLAS Integration**: Optimized linear algebra operations

### **Memory Usage Guidelines**
- **Lists**: ~8 bytes per element + 8 bytes overhead
- **Dictionaries**: ~30 bytes per key-value pair + 8 bytes overhead
- **NumPy Arrays**: 4-8 bytes per element + fixed overhead
- **pandas DataFrames**: ~100-200 bytes per cell + index overhead

---

## 🔗 External API References

### **Financial Data APIs**
- **Alpha Vantage**: Stock market data and technical indicators
- **IEX Cloud**: Real-time and historical market data
- **Financial Modeling Prep**: Financial statements and ratios
- **Yahoo Finance**: Free historical price data

### **Python Libraries**
- **NumPy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing and statistical functions
- **Matplotlib**: Data visualization and plotting
- **scikit-learn**: Machine learning and predictive modeling

### **Data Formats**
- **CSV**: Comma-separated values for tabular data
- **JSON**: JavaScript Object Notation for structured data
- **XML**: Extensible Markup Language for complex data
- **Parquet**: Columnar storage format for big data

---

## 📝 Configuration Parameters

### **Default Settings**
```python
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual
DEFAULT_MARKET_RISK_PREMIUM = 0.06  # 6% annual
DEFAULT_INFLATION_RATE = 0.03  # 3% annual
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence
DEFAULT_SIMULATION_PATHS = 10000  # Monte Carlo paths
DEFAULT_VOLATILITY_WINDOW = 252  # Trading days
```

### **TVM Parameters**
```python
COMPOUNDING_FREQUENCIES = {
    'annual': 1,
    'semiannual': 2,
    'quarterly': 4,
    'monthly': 12,
    'daily': 365,
    'continuous': float('inf')
}

ANNUITY_TYPES = {
    'ordinary': 0,  # Payments at end of period
    'due': 1        # Payments at beginning of period
}
```

### **Risk Parameters**
```python
VAR_METHODS = ['historical', 'parametric', 'monte_carlo']
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99, 0.999]
STRESS_TEST_SCENARIOS = {
    'market_crash': {'return': -0.30, 'volatility': 2.0},
    'recession': {'return': -0.15, 'volatility': 1.5},
    'normal': {'return': 0.08, 'volatility': 1.0}
}
```

---

## 🧪 Testing Framework

### **Unit Test Structure**
```python
import unittest
from financial_utils import TimeValueMoney

class TestTVMCalculations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tvm = TimeValueMoney()

    def test_future_value_single(self):
        """Test future value calculation."""
        result = self.tvm.future_value_single(1000, 0.05, 1)
        expected = 1050.0
        self.assertAlmostEqual(result, expected, places=2)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        with self.assertRaises(ValueError):
            self.tvm.future_value_single(-1000, 0.05, 1)

    def tearDown(self):
        """Clean up after tests."""
        pass
```

### **Integration Test Example**
```python
def test_portfolio_workflow():
    """Test complete portfolio workflow."""
    # Create portfolio
    portfolio = create_portfolio()
    initial_value = calculate_portfolio_value(portfolio)

    # Simulate market changes
    update_prices(portfolio)
    final_value = calculate_portfolio_value(portfolio)

    # Verify calculations
    assert final_value != initial_value
    assert all(position['value'] >= 0 for position in portfolio.values())
```

---

## 📋 Glossary

### **Financial Terms**
- **PV (Present Value)**: Current worth of future cash flows
- **FV (Future Value)**: Value of current investment at future date
- **PMT (Payment)**: Regular payment amount
- **IRR (Internal Rate of Return)**: Discount rate that makes NPV zero
- **NPV (Net Present Value)**: Difference between PV of inflows and outflows
- **VaR (Value at Risk)**: Maximum expected loss at confidence level
- **Beta**: Measure of systematic risk relative to market
- **Alpha**: Excess return above expected given beta

### **Technical Terms**
- **API (Application Programming Interface)**: Interface for software interaction
- **SDK (Software Development Kit)**: Tools for building applications
- **REST (Representational State Transfer)**: Web service architecture
- **JSON (JavaScript Object Notation)**: Data interchange format
- **CSV (Comma-Separated Values)**: Tabular data format
- **OOP (Object-Oriented Programming)**: Programming paradigm using objects
- **SOLID**: Object-oriented design principles
- **DRY (Don't Repeat Yourself)**: Software development principle

### **Statistical Terms**
- **Mean**: Arithmetic average of values
- **Median**: Middle value in ordered dataset
- **Mode**: Most frequently occurring value
- **Standard Deviation**: Measure of data dispersion
- **Skewness**: Measure of data asymmetry
- **Kurtosis**: Measure of data tailedness
- **p-value**: Probability of observing result by chance
- **Confidence Interval**: Range containing true parameter with confidence

---

*This API reference provides comprehensive documentation for all utilities in the Quantitative Finance Learning Platform. Use this guide to understand function signatures, parameters, return values, and implementation details.*

**Made with ❤️ by MeridianAlgo**
