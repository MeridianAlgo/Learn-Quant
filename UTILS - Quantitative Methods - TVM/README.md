# Time Value of Money (TVM) - Core Financial Calculations

## 📋 Overview

This utility provides comprehensive Time Value of Money (TVM) calculations essential for financial analysis, investment evaluation, and capital budgeting. TVM is the foundation of quantitative finance and corporate finance.

## 🎯 Key Concepts

### **Time Value of Money Principles**
- **Present Value (PV)**: Current worth of future cash flows
- **Future Value (FV)**: Value of current investment at future date
- **Interest**: Compensation for use of money over time
- **Compounding**: Interest earned on interest
- **Discounting**: Process of determining present value

### **TVM Components**
- **PV**: Present Value
- **FV**: Future Value
- **PMT**: Payment/Installment amount
- **i**: Interest rate per period
- **n**: Number of compounding periods
- **Type**: Ordinary annuity (0) or annuity due (1)

### **Financial Applications**
- **Investment Valuation**: Stock, bond, and project evaluation
- **Loan Analysis**: Mortgage, auto loan, and personal loan calculations
- **Retirement Planning**: 401(k), IRA, and pension calculations
- **Capital Budgeting**: NPV, IRR, and payback period analysis
- **Bond Pricing**: Yield calculations and duration analysis

## 💻 Implementation

### **Basic TVM Calculations**
```python
# Future Value of a Single Sum
def future_value_single(pv: float, i: float, n: int) -> float:
    """Calculate future value of a single present value."""
    return pv * (1 + i) ** n

# Present Value of a Single Sum
def present_value_single(fv: float, i: float, n: int) -> float:
    """Calculate present value of a single future value."""
    return fv / (1 + i) ** n

# Ordinary Annuity Future Value
def future_value_annuity(pmt: float, i: float, n: int) -> float:
    """Calculate future value of an ordinary annuity."""
    return pmt * ((1 + i) ** n - 1) / i

# Ordinary Annuity Present Value
def present_value_annuity(pmt: float, i: float, n: int) -> float:
    """Calculate present value of an ordinary annuity."""
    return pmt * (1 - 1 / (1 + i) ** n) / i

# Annuity Payment
def annuity_payment(pv: float, i: float, n: int) -> float:
    """Calculate required payment for a loan or annuity."""
    return pv * i / (1 - 1 / (1 + i) ** n)
```

### **Advanced TVM Calculations**
```python
# Net Present Value (NPV)
def calculate_npv(cash_flows: list, discount_rate: float) -> float:
    """Calculate Net Present Value of a series of cash flows."""
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate) ** t
    return npv

# Internal Rate of Return (IRR)
def calculate_irr(cash_flows: list, initial_guess: float = 0.1) -> float:
    """Calculate Internal Rate of Return using iterative method."""
    # Simplified IRR calculation using numpy
    from scipy.optimize import newton

    def npv_function(rate):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))

    try:
        return newton(npv_function, initial_guess)
    except:
        return None

# Effective Annual Rate (EAR)
def effective_annual_rate(nominal_rate: float, compounding_freq: int) -> float:
    """Calculate effective annual rate from nominal rate."""
    return (1 + nominal_rate / compounding_freq) ** compounding_freq - 1

# Continuous Compounding
def continuous_compounding(pv: float, rate: float, time: float) -> float:
    """Calculate future value with continuous compounding."""
    import math
    return pv * math.exp(rate * time)
```

### **Loan and Mortgage Calculations**
```python
# Loan Payment Calculation
def calculate_loan_payment(principal: float, annual_rate: float,
                          years: int, payments_per_year: int = 12) -> float:
    """Calculate monthly loan payment."""
    periodic_rate = annual_rate / payments_per_year
    total_payments = years * payments_per_year
    return principal * (periodic_rate * (1 + periodic_rate) ** total_payments) / \
           ((1 + periodic_rate) ** total_payments - 1)

# Amortization Schedule
def generate_amortization_schedule(principal: float, annual_rate: float,
                                  years: int, payments_per_year: int = 12) -> list:
    """Generate loan amortization schedule."""
    monthly_payment = calculate_loan_payment(principal, annual_rate, years, payments_per_year)
    periodic_rate = annual_rate / payments_per_year

    balance = principal
    schedule = []

    for period in range(1, years * payments_per_year + 1):
        interest_payment = balance * periodic_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment

        schedule.append({
            'period': period,
            'payment': monthly_payment,
            'principal': principal_payment,
            'interest': interest_payment,
            'balance': max(0, balance)  # Avoid negative balance due to rounding
        })

    return schedule
```

## 📊 Examples

### **Example 1: Investment Analysis**
```python
def investment_analysis():
    """Comprehensive investment analysis using TVM."""
    print("=== Investment Analysis ===")

    # Scenario: Investing $10,000 for 10 years
    principal = 10000
    annual_return = 0.08  # 8% annual return
    years = 10

    # Future value with annual compounding
    fv_annual = future_value_single(principal, annual_return, years)
    print(f"Future Value (Annual Compounding): ${fv_annual".2f"}")

    # Future value with monthly compounding
    monthly_rate = annual_return / 12
    fv_monthly = future_value_single(principal, monthly_rate, years * 12)
    print(f"Future Value (Monthly Compounding): ${fv_monthly".2f"}")

    # Effective annual rate
    ear = effective_annual_rate(annual_return, 12)
    print(f"Effective Annual Rate: {ear".4f"}")

    # Continuous compounding
    fv_continuous = continuous_compounding(principal, annual_return, years)
    print(f"Future Value (Continuous Compounding): ${fv_continuous".2f"}")

    print()

def investment_analysis():
    """Comprehensive investment analysis using TVM."""
    print("=== Investment Analysis ===")

    # Scenario: Investing $10,000 for 10 years
    principal = 10000
    annual_return = 0.08  # 8% annual return
    years = 10

    # Future value with annual compounding
    fv_annual = future_value_single(principal, annual_return, years)
    print(f"Future Value (Annual Compounding): ${fv_annual".2f"}")

    # Future value with monthly compounding
    monthly_rate = annual_return / 12
    fv_monthly = future_value_single(principal, monthly_rate, years * 12)
    print(f"Future Value (Monthly Compounding): ${fv_monthly".2f"}")

    # Effective annual rate
    ear = effective_annual_rate(annual_return, 12)
    print(f"Effective Annual Rate: {ear".4f"}")

    # Continuous compounding
    fv_continuous = continuous_compounding(principal, annual_return, years)
    print(f"Future Value (Continuous Compounding): ${fv_continuous".2f"}")

    print()
```

### **Example 2: Loan Analysis**
```python
def loan_analysis():
    """Comprehensive loan analysis."""
    print("=== Loan Analysis ===")

    # Mortgage parameters
    loan_amount = 300000  # $300,000 mortgage
    annual_rate = 0.045   # 4.5% annual interest rate
    loan_term = 30        # 30 years
    payments_per_year = 12

    # Calculate monthly payment
    monthly_payment = calculate_loan_payment(loan_amount, annual_rate, loan_term, payments_per_year)
    print(f"Monthly Payment: ${monthly_payment".2f"}")

    # Generate amortization schedule
    schedule = generate_amortization_schedule(loan_amount, annual_rate, loan_term, payments_per_year)

    # Show first 12 payments
    print("\nFirst 12 Payments:")
    print(f"{'Period'"<6"} {'Payment'"<10"} {'Principal'"<10"} {'Interest'"<10"} {'Balance'"<12"}")
    print("-" * 60)
    for payment in schedule[:12]:
        print(f"{payment['period']"<6"} {payment['payment']"<10.2f"} {payment['principal']"<10.2f"} {payment['interest']"<10.2f"} {payment['balance']"<12.2f"}")

    # Total interest paid
    total_interest = sum(payment['interest'] for payment in schedule)
    print(f"\nTotal Interest Paid: ${total_interest".2f"}")
    print(f"Total Amount Paid: ${total_interest + loan_amount".2f"}")

    print()
```

### **Example 3: Retirement Planning**
```python
def retirement_planning():
    """Retirement planning calculations."""
    print("=== Retirement Planning ===")

    # Current age and retirement goals
    current_age = 30
    retirement_age = 65
    years_to_retirement = retirement_age - current_age

    # Financial goals
    desired_annual_income = 60000  # $60,000 annual retirement income
    life_expectancy = 90
    years_in_retirement = life_expectancy - retirement_age

    # Assumptions
    annual_return = 0.07  # 7% annual return
    inflation_rate = 0.03  # 3% annual inflation
    current_savings = 25000  # $25,000 already saved

    # Calculate future value of current savings
    future_value_savings = future_value_single(current_savings, annual_return, years_to_retirement)
    print(f"Future Value of Current Savings: ${future_value_savings".2f"}")

    # Calculate required annual contribution
    # Present value of retirement income stream
    pv_retirement_income = present_value_annuity(desired_annual_income, annual_return, years_in_retirement)
    print(f"PV of Required Retirement Income: ${pv_retirement_income".2f"}")

    # Amount needed at retirement (adjusted for inflation)
    inflation_adjusted_pv = pv_retirement_income * (1 + inflation_rate) ** years_to_retirement
    print(f"Inflation-Adjusted Amount Needed: ${inflation_adjusted_pv".2f"}")

    # Annual contribution required
    contribution_needed = (inflation_adjusted_pv - future_value_savings) * annual_return / \
                          ((1 + annual_return) ** years_to_retirement - 1)
    print(f"Required Annual Contribution: ${contribution_needed".2f"}")

    # Sensitivity analysis
    print("\nSensitivity Analysis:")
    for return_rate in [0.05, 0.07, 0.09]:
        required_contrib = (inflation_adjusted_pv - future_value_single(current_savings, return_rate, years_to_retirement)) * return_rate / \
                          ((1 + return_rate) ** years_to_retirement - 1)
        print(f"At {return_rate".1%"} return: ${required_contrib".2f"} annual contribution")

    print()
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_tvm.py -v
```

## 📚 References

- [CFA Institute - Time Value of Money](https://www.cfainstitute.org/)
- [Corporate Finance Institute - TVM](https://corporatefinanceinstitute.com/)
- [Investopedia - Time Value of Money](https://www.investopedia.com/terms/t/timevalueofmoney.asp)
- [Python Financial Calculations](https://www.oreilly.com/library/view/python-for-finance/9781492024326/)

## 🎓 Learning Path

### **Prerequisites**
- Basic mathematics (exponents, algebra)
- Understanding of financial markets

### **Next Steps**
- **Capital Budgeting**: NPV, IRR, and project evaluation
- **Bond Valuation**: Fixed income securities pricing
- **Statistics**: Probability and statistical analysis

### **Assessment**
1. Calculate the future value of $10,000 invested for 20 years at 6% annual interest
2. Determine the monthly payment for a $250,000 mortgage at 4% for 30 years
3. Calculate the present value of an annuity that pays $5,000 quarterly for 15 years at 5%
4. Build a retirement calculator that accounts for inflation and different return scenarios

---

*This utility provides the foundation for all quantitative finance calculations. Master TVM to understand the fundamental relationship between money and time.*
