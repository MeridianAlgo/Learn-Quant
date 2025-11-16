# 💡 Interactive Examples - Practical Applications

## 🎯 Real-World Financial Applications

This section contains practical examples and case studies demonstrating how to apply quantitative finance concepts to real-world problems.

---

## 📊 Example Projects

### **Example 1: Personal Portfolio Tracker**

#### **Project Description**
Build a comprehensive personal portfolio tracker that monitors your investments, calculates performance metrics, and provides insights for better investment decisions.

#### **Key Features**
- Portfolio position tracking
- Performance calculation and visualization
- Risk metrics (VaR, Sharpe ratio, max drawdown)
- Transaction logging and history
- Portfolio rebalancing recommendations

#### **Code Implementation**
```python
# portfolio_tracker.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PortfolioTracker:
    def __init__(self):
        self.positions = {}  # ticker: {'shares': int, 'avg_cost': float}
        self.transactions = []  # List of transaction records
        self.price_history = {}  # ticker: list of (date, price) tuples

    def add_position(self, ticker, shares, price, date=None):
        """Add or update a position."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        if ticker in self.positions:
            # Update existing position
            existing_shares = self.positions[ticker]['shares']
            existing_cost = self.positions[ticker]['avg_cost']
            total_shares = existing_shares + shares
            total_cost = (existing_shares * existing_cost + shares * price)
            self.positions[ticker]['avg_cost'] = total_cost / total_shares
            self.positions[ticker]['shares'] = total_shares
        else:
            # New position
            self.positions[ticker] = {'shares': shares, 'avg_cost': price}

        # Log transaction
        self.transactions.append({
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'date': date,
            'type': 'BUY'
        })

    def update_prices(self, price_data):
        """Update current prices for all positions."""
        for ticker, price in price_data.items():
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            self.price_history[ticker].append((datetime.now().strftime('%Y-%m-%d'), price))

    def calculate_portfolio_value(self):
        """Calculate current portfolio value."""
        total_value = 0
        for ticker, position in self.positions.items():
            # Get latest price (simplified - in real app, fetch from API)
            if ticker in self.price_history and self.price_history[ticker]:
                current_price = self.price_history[ticker][-1][1]
                position_value = position['shares'] * current_price
                total_value += position_value
        return total_value

    def calculate_performance(self):
        """Calculate portfolio performance metrics."""
        portfolio_values = []
        dates = []

        # Calculate historical portfolio values
        all_dates = set()
        for ticker, history in self.price_history.items():
            for date, price in history:
                all_dates.add(date)

        sorted_dates = sorted(all_dates)

        for date in sorted_dates:
            daily_value = 0
            for ticker, position in self.positions.items():
                if ticker in self.price_history:
                    # Find price for this date (simplified)
                    ticker_history = self.price_history[ticker]
                    date_prices = [price for d, price in ticker_history if d <= date]
                    if date_prices:
                        price = date_prices[-1]
                        daily_value += position['shares'] * price
            if daily_value > 0:
                portfolio_values.append(daily_value)
                dates.append(date)

        if not portfolio_values:
            return {}

        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        return {
            'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'annualized_return': np.mean(returns) * 252,
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': (np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(portfolio_values),
            'portfolio_values': portfolio_values,
            'dates': dates
        }

    def calculate_max_drawdown(self, values):
        """Calculate maximum drawdown."""
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def generate_report(self):
        """Generate comprehensive portfolio report."""
        current_value = self.calculate_portfolio_value()
        performance = self.calculate_performance()

        print("=== Portfolio Report ===")
        print(f"Current Value: ${current_value",.2f"}")

        if performance:
            print(f"Total Return: {performance['total_return']".2%"}")
            print(f"Annualized Return: {performance['annualized_return']".2%"}")
            print(f"Volatility: {performance['volatility']".2%"}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']".2f"}")
            print(f"Max Drawdown: {performance['max_drawdown']".2%"}")

        print("\n=== Holdings ===")
        for ticker, position in self.positions.items():
            current_price = self.price_history.get(ticker, [('', 0)])[-1][1]
            current_value = position['shares'] * current_price
            pnl = (current_price - position['avg_cost']) / position['avg_cost']
            print(f"{ticker}: {position['shares']} shares @ ${position['avg_cost']".2f"}")
            print(f"  Current: ${current_price".2f"}, Value: ${current_value".2f"}, P&L: {pnl".2%"}")

# Example usage
tracker = PortfolioTracker()

# Add some positions
tracker.add_position('AAPL', 100, 150.25, '2024-01-01')
tracker.add_position('GOOGL', 10, 2800.50, '2024-01-01')
tracker.add_position('MSFT', 50, 350.75, '2024-01-01')

# Update prices (simulate price changes)
price_updates = {
    'AAPL': 165.50,
    'GOOGL': 2950.25,
    'MSFT': 375.00
}
tracker.update_prices(price_updates)

# Generate report
tracker.generate_report()
```

#### **Learning Outcomes**
- Portfolio position tracking and management
- Performance calculation and analysis
- Risk metrics computation
- Data visualization and reporting
- Object-oriented programming for financial applications

---

### **Example 2: Monte Carlo Option Pricer**

#### **Project Description**
Build a Monte Carlo simulation engine for pricing European options, including Greeks calculation and risk analysis.

#### **Key Features**
- Black-Scholes model implementation
- Monte Carlo simulation engine
- Option Greeks calculation
- Volatility modeling
- Risk analysis and stress testing

#### **Code Implementation**
```python
# monte_carlo_option_pricer.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

class MonteCarloOptionPricer:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def monte_carlo_price(self, S, K, T, r, sigma, option_type='call',
                          n_simulations=100000):
        """Price option using Monte Carlo simulation."""
        # Generate random normal variables
        Z = np.random.standard_normal(n_simulations)

        # Simulate stock prices at maturity
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Discount payoffs to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)

        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(n_simulations)

        return option_price, std_error

    def calculate_greeks(self, S, K, T, r, sigma, option_type='call',
                        n_simulations=100000, epsilon=0.01):
        """Calculate option Greeks using finite differences."""
        # Price at current parameters
        price, _ = self.monte_carlo_price(S, K, T, r, sigma, option_type, n_simulations)

        # Delta (change in stock price)
        price_up, _ = self.monte_carlo_price(S + epsilon, K, T, r, sigma, option_type, n_simulations)
        price_down, _ = self.monte_carlo_price(S - epsilon, K, T, r, sigma, option_type, n_simulations)
        delta = (price_up - price_down) / (2 * epsilon)

        # Gamma (second derivative)
        gamma = (price_up - 2 * price + price_down) / (epsilon ** 2)

        # Vega (change in volatility)
        price_vol_up, _ = self.monte_carlo_price(S, K, T, r, sigma + epsilon, option_type, n_simulations)
        price_vol_down, _ = self.monte_carlo_price(S, K, T, r, sigma - epsilon, option_type, n_simulations)
        vega = (price_vol_up - price_vol_down) / (2 * epsilon)

        # Theta (time decay)
        price_time_up, _ = self.monte_carlo_price(S, K, T + epsilon, r, sigma, option_type, n_simulations)
        theta = (price_time_up - price) / epsilon

        # Rho (interest rate sensitivity)
        price_rate_up, _ = self.monte_carlo_price(S, K, T, r + epsilon, sigma, option_type, n_simulations)
        price_rate_down, _ = self.monte_carlo_price(S, K, T, r - epsilon, sigma, option_type, n_simulations)
        rho = (price_rate_up - price_rate_down) / (2 * epsilon)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def volatility_surface_analysis(self, S, K, T, r, option_type='call',
                                   strikes=None, maturities=None):
        """Analyze volatility surface."""
        if strikes is None:
            strikes = np.linspace(0.8 * S, 1.2 * S, 10)
        if maturities is None:
            maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

        volatility_surface = np.zeros((len(maturities), len(strikes)))
        prices = np.zeros((len(maturities), len(strikes)))

        for i, T_i in enumerate(maturities):
            for j, K_j in enumerate(strikes):
                # Implied volatility calculation (simplified)
                price = self.black_scholes_price(S, K_j, T_i, r, 0.2, option_type)
                prices[i, j] = price

                # For this example, we'll use a simple volatility model
                # In practice, you'd solve for implied volatility
                volatility_surface[i, j] = 0.2 + 0.1 * (K_j / S - 1) ** 2 + 0.05 * T_i

        return strikes, maturities, prices, volatility_surface

    def run_comprehensive_analysis(self, S=100, K=100, T=1, r=0.05, sigma=0.2,
                                  option_type='call', n_simulations=50000):
        """Run comprehensive option analysis."""
        print("=== Monte Carlo Option Analysis ===")
        print(f"Stock Price: ${S}")
        print(f"Strike Price: ${K}")
        print(f"Time to Maturity: {T} years")
        print(f"Risk-free Rate: {r".2%"}")
        print(f"Volatility: {sigma".2%"}")
        print(f"Option Type: {option_type.upper()}")
        print(f"Number of Simulations: {n_simulations","}")
        print("-" * 50)

        # Time the calculations
        start_time = time.time()

        # Black-Scholes price
        bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
        print(f"Black-Scholes Price: ${bs_price".4f"}")

        # Monte Carlo price
        mc_price, std_error = self.monte_carlo_price(S, K, T, r, sigma, option_type, n_simulations)
        print(f"Monte Carlo Price: ${mc_price".4f"} ± ${std_error".4f"}")

        # Calculate Greeks
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type, n_simulations)
        print("
Greeks:")
        print(f"Delta: {greeks['delta']".4f"}")
        print(f"Gamma: {greeks['gamma']".4f"}")
        print(f"Vega: {greeks['vega']".4f"}")
        print(f"Theta: {greeks['theta']".4f"}")
        print(f"Rho: {greeks['rho']".4f"}")

        # Volatility surface
        strikes, maturities, prices, vol_surface = self.volatility_surface_analysis(S, K, T, r, option_type)
        print(f"\nVolatility Surface Analysis: {len(strikes)} strikes, {len(maturities)} maturities")

        end_time = time.time()
        print(f"\nAnalysis completed in {end_time - start_time".2f"} seconds")

        return {
            'black_scholes_price': bs_price,
            'monte_carlo_price': mc_price,
            'std_error': std_error,
            'greeks': greeks,
            'volatility_surface': vol_surface
        }

# Example usage
pricer = MonteCarloOptionPricer()
results = pricer.run_comprehensive_analysis()
```

#### **Learning Outcomes**
- Monte Carlo simulation techniques
- Options pricing and Greeks calculation
- Volatility modeling and analysis
- Risk management applications
- Performance optimization for simulations

---

### **Example 3: Financial Statement Analyzer**

#### **Project Description**
Create a comprehensive financial statement analyzer that processes income statements, balance sheets, and cash flow statements to calculate key ratios and metrics.

#### **Key Features**
- Financial statement parsing and validation
- Ratio calculation and analysis
- Trend analysis and comparison
- Investment recommendation engine
- Report generation and visualization

#### **Code Implementation**
```python
# financial_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FinancialStatementAnalyzer:
    def __init__(self):
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
        self.ratios = {}

    def load_financial_statements(self, income_data: Dict, balance_data: Dict, cash_data: Dict):
        """Load financial statement data."""
        self.income_statement = pd.DataFrame(income_data)
        self.balance_sheet = pd.DataFrame(balance_data)
        self.cash_flow = pd.DataFrame(cash_data)

    def calculate_profitability_ratios(self) -> Dict[str, float]:
        """Calculate profitability ratios."""
        if self.income_statement is None:
            return {}

        revenue = self.income_statement.get('Revenue', [0]).iloc[-1]
        gross_profit = self.income_statement.get('Gross Profit', [0]).iloc[-1]
        operating_income = self.income_statement.get('Operating Income', [0]).iloc[-1]
        net_income = self.income_statement.get('Net Income', [0]).iloc[-1]
        total_assets = self.balance_sheet.get('Total Assets', [0]).iloc[-1] if self.balance_sheet is not None else 1
        equity = self.balance_sheet.get('Total Equity', [0]).iloc[-1] if self.balance_sheet is not None else 1

        ratios = {}
        if revenue > 0:
            ratios['Gross Margin'] = gross_profit / revenue
            ratios['Operating Margin'] = operating_income / revenue
            ratios['Net Margin'] = net_income / revenue

        if total_assets > 0:
            ratios['ROA'] = net_income / total_assets

        if equity > 0:
            ratios['ROE'] = net_income / equity

        ratios['EPS'] = net_income / 1000000  # Assuming 1M shares

        return ratios

    def calculate_liquidity_ratios(self) -> Dict[str, float]:
        """Calculate liquidity ratios."""
        if self.balance_sheet is None:
            return {}

        current_assets = self.balance_sheet.get('Current Assets', [0]).iloc[-1]
        current_liabilities = self.balance_sheet.get('Current Liabilities', [0]).iloc[-1]
        inventory = self.balance_sheet.get('Inventory', [0]).iloc[-1]
        cash = self.balance_sheet.get('Cash', [0]).iloc[-1]

        ratios = {}
        if current_liabilities > 0:
            ratios['Current Ratio'] = current_assets / current_liabilities
            ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities
            ratios['Cash Ratio'] = cash / current_liabilities

        return ratios

    def calculate_solvency_ratios(self) -> Dict[str, float]:
        """Calculate solvency ratios."""
        if self.balance_sheet is None:
            return {}

        total_assets = self.balance_sheet.get('Total Assets', [0]).iloc[-1]
        total_liabilities = self.balance_sheet.get('Total Liabilities', [0]).iloc[-1]
        long_term_debt = self.balance_sheet.get('Long Term Debt', [0]).iloc[-1]
        equity = self.balance_sheet.get('Total Equity', [0]).iloc[-1]

        ratios = {}
        if total_assets > 0:
            ratios['Debt to Assets'] = total_liabilities / total_assets
            ratios['Financial Leverage'] = total_assets / equity

        if equity > 0:
            ratios['Debt to Equity'] = total_liabilities / equity

        ratios['Interest Coverage'] = self.income_statement.get('EBIT', [0]).iloc[-1] / long_term_debt if long_term_debt > 0 else float('inf')

        return ratios

    def calculate_efficiency_ratios(self) -> Dict[str, float]:
        """Calculate efficiency ratios."""
        if self.income_statement is None or self.balance_sheet is None:
            return {}

        revenue = self.income_statement.get('Revenue', [0]).iloc[-1]
        total_assets = self.balance_sheet.get('Total Assets', [0]).iloc[-1]
        fixed_assets = self.balance_sheet.get('Fixed Assets', [0]).iloc[-1]
        inventory = self.balance_sheet.get('Inventory', [0]).iloc[-1]
        receivables = self.balance_sheet.get('Accounts Receivable', [0]).iloc[-1]

        ratios = {}
        if total_assets > 0:
            ratios['Asset Turnover'] = revenue / total_assets

        if fixed_assets > 0:
            ratios['Fixed Asset Turnover'] = revenue / fixed_assets

        if inventory > 0:
            cogs = self.income_statement.get('COGS', [0]).iloc[-1]
            ratios['Inventory Turnover'] = cogs / inventory

        if receivables > 0:
            ratios['Receivables Turnover'] = revenue / receivables

        return ratios

    def calculate_valuation_ratios(self) -> Dict[str, float]:
        """Calculate valuation ratios."""
        if self.income_statement is None or self.balance_sheet is None:
            return {}

        stock_price = 100  # Placeholder - in real app, fetch from API
        shares_outstanding = 1000000  # Placeholder
        net_income = self.income_statement.get('Net Income', [0]).iloc[-1]
        book_value = self.balance_sheet.get('Total Equity', [0]).iloc[-1]
        revenue = self.income_statement.get('Revenue', [0]).iloc[-1]

        market_cap = stock_price * shares_outstanding

        ratios = {}
        ratios['P/E Ratio'] = stock_price / (net_income / shares_outstanding) if net_income > 0 else float('inf')
        ratios['P/B Ratio'] = market_cap / book_value if book_value > 0 else float('inf')
        ratios['P/S Ratio'] = market_cap / revenue if revenue > 0 else float('inf')

        return ratios

    def analyze_company(self) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive company analysis."""
        self.ratios = {
            'Profitability': self.calculate_profitability_ratios(),
            'Liquidity': self.calculate_liquidity_ratios(),
            'Solvency': self.calculate_solvency_ratios(),
            'Efficiency': self.calculate_efficiency_ratios(),
            'Valuation': self.calculate_valuation_ratios()
        }

        return self.ratios

    def generate_investment_recommendation(self) -> str:
        """Generate investment recommendation based on ratios."""
        if not self.ratios:
            return "Insufficient data for recommendation"

        # Simplified scoring system
        score = 0

        # Profitability (40% weight)
        profitability = self.ratios['Profitability']
        if profitability.get('ROE', 0) > 0.15: score += 20
        elif profitability.get('ROE', 0) > 0.10: score += 15
        elif profitability.get('ROE', 0) > 0.05: score += 10

        if profitability.get('Net Margin', 0) > 0.10: score += 20
        elif profitability.get('Net Margin', 0) > 0.05: score += 15

        # Liquidity (20% weight)
        liquidity = self.ratios['Liquidity']
        if liquidity.get('Current Ratio', 0) > 2: score += 15
        elif liquidity.get('Current Ratio', 0) > 1.5: score += 10

        # Solvency (20% weight)
        solvency = self.ratios['Solvency']
        if solvency.get('Debt to Equity', 0) < 0.5: score += 15
        elif solvency.get('Debt to Equity', 0) < 1.0: score += 10

        # Efficiency (20% weight)
        efficiency = self.ratios['Efficiency']
        if efficiency.get('Asset Turnover', 0) > 1.0: score += 10
        elif efficiency.get('Asset Turnover', 0) > 0.5: score += 5

        # Recommendation
        if score >= 70:
            return "Strong Buy"
        elif score >= 55:
            return "Buy"
        elif score >= 40:
            return "Hold"
        elif score >= 25:
            return "Weak Hold"
        else:
            return "Sell"

    def print_analysis_report(self):
        """Print comprehensive analysis report."""
        if not self.ratios:
            self.analyze_company()

        print("=== Financial Statement Analysis Report ===")
        print("-" * 50)

        for category, ratios in self.ratios.items():
            print(f"\n{category} Ratios:")
            print("-" * 30)
            for ratio_name, value in ratios.items():
                if isinstance(value, float):
                    if 'Ratio' in ratio_name or 'Margin' in ratio_name or 'RO' in ratio_name:
                        print(f"{ratio_name}: {value".3f"}")
                    else:
                        print(f"{ratio_name}: {value".2f"}")
                else:
                    print(f"{ratio_name}: {value}")

        recommendation = self.generate_investment_recommendation()
        print(f"\nInvestment Recommendation: {recommendation}")

# Example usage
analyzer = FinancialStatementAnalyzer()

# Sample financial data (in millions)
income_data = {
    'Year': [2022, 2023, 2024],
    'Revenue': [10000, 12000, 14000],
    'COGS': [6000, 7200, 8400],
    'Gross Profit': [4000, 4800, 5600],
    'Operating Expenses': [2000, 2400, 2800],
    'Operating Income': [2000, 2400, 2800],
    'Interest Expense': [200, 180, 160],
    'EBIT': [2200, 2580, 2960],
    'Net Income': [1600, 1920, 2240]
}

balance_data = {
    'Year': [2022, 2023, 2024],
    'Cash': [2000, 2500, 3000],
    'Accounts Receivable': [1500, 1800, 2100],
    'Inventory': [1000, 1200, 1400],
    'Current Assets': [4500, 5500, 6500],
    'Fixed Assets': [8000, 9000, 10000],
    'Total Assets': [12500, 14500, 16500],
    'Current Liabilities': [3000, 3500, 4000],
    'Long Term Debt': [4000, 3800, 3600],
    'Total Liabilities': [7000, 7300, 7600],
    'Total Equity': [5500, 7200, 8900]
}

cash_data = {
    'Year': [2022, 2023, 2024],
    'Operating Cash Flow': [2500, 3000, 3500],
    'Investing Cash Flow': [-1500, -1800, -2100],
    'Financing Cash Flow': [-800, -600, -400],
    'Net Cash Flow': [200, 600, 1000]
}

analyzer.load_financial_statements(income_data, balance_data, cash_data)
analyzer.analyze_company()
analyzer.print_analysis_report()
```

#### **Learning Outcomes**
- Financial statement analysis and interpretation
- Ratio calculation and benchmarking
- Investment decision-making frameworks
- Financial modeling and valuation
- Report generation and presentation

---

## 🎯 Assessment and Next Steps

### **Self-Assessment Questions**
1. **Portfolio Tracker**: Can you track multiple assets and calculate performance metrics?
2. **Option Pricer**: Can you price options using both Black-Scholes and Monte Carlo methods?
3. **Financial Analyzer**: Can you calculate and interpret key financial ratios?

### **Extension Ideas**
- **Real-Time Data**: Connect to financial APIs for live data
- **Machine Learning**: Add predictive models for price forecasting
- **Web Interface**: Create a web-based dashboard
- **Advanced Analytics**: Implement more sophisticated risk models

### **Portfolio Projects**
- **Personal Finance App**: Complete personal finance management system
- **Trading Simulator**: Paper trading with real-time market data
- **Investment Research Platform**: Comprehensive stock analysis tool
- **Risk Management System**: Enterprise-level risk monitoring

---

*These examples demonstrate practical applications of quantitative finance concepts. Each project builds on the utilities in this repository and provides a foundation for real-world financial applications.*

**Made with ❤️ by MeridianAlgo**
