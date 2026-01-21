"""
Time Value of Money (TVM) Utility - Core Financial Calculations

This module provides comprehensive TVM calculations for:
- Investment analysis and valuation
- Loan and mortgage calculations
- Retirement planning
- Capital budgeting
- Bond and fixed income analysis

Author: MeridianAlgo
Version: 1.0.0
"""

import math
from typing import Dict, List, Optional

from scipy.optimize import newton


class TimeValueMoney:
    """
    Comprehensive Time Value of Money calculations.

    This class provides methods for:
    - Present and future value calculations
    - Annuity and perpetuity calculations
    - Loan and mortgage analysis
    - Investment analysis
    - Capital budgeting
    """

    def __init__(self):
        """Initialize the TimeValueMoney class."""
        pass

    def future_value_single(self, pv: float, rate: float, periods: int, compounding: int = 1) -> float:
        """
        Calculate future value of a single present value.

        Args:
            pv: Present value
            rate: Interest rate per period
            periods: Number of periods
            compounding: Compounding frequency per period

        Returns:
            float: Future value
        """
        effective_rate = rate / compounding
        effective_periods = periods * compounding
        return pv * (1 + effective_rate) ** effective_periods

    def present_value_single(self, fv: float, rate: float, periods: int, compounding: int = 1) -> float:
        """
        Calculate present value of a single future value.

        Args:
            fv: Future value
            rate: Interest rate per period
            periods: Number of periods
            compounding: Compounding frequency per period

        Returns:
            float: Present value
        """
        effective_rate = rate / compounding
        effective_periods = periods * compounding
        return fv / (1 + effective_rate) ** effective_periods

    def future_value_annuity(self, pmt: float, rate: float, periods: int, type: int = 0, compounding: int = 1) -> float:
        """
        Calculate future value of an ordinary annuity.

        Args:
            pmt: Payment per period
            rate: Interest rate per period
            periods: Number of periods
            type: 0 for ordinary annuity, 1 for annuity due
            compounding: Compounding frequency per period

        Returns:
            float: Future value of annuity
        """
        effective_rate = rate / compounding
        effective_periods = periods * compounding

        if type == 0:  # Ordinary annuity
            fv = pmt * ((1 + effective_rate) ** effective_periods - 1) / effective_rate
        else:  # Annuity due
            fv = pmt * ((1 + effective_rate) ** effective_periods - 1) / effective_rate * (1 + effective_rate)

        return fv

    def present_value_annuity(
        self, pmt: float, rate: float, periods: int, type: int = 0, compounding: int = 1
    ) -> float:
        """
        Calculate present value of an ordinary annuity.

        Args:
            pmt: Payment per period
            rate: Interest rate per period
            periods: Number of periods
            type: 0 for ordinary annuity, 1 for annuity due
            compounding: Compounding frequency per period

        Returns:
            float: Present value of annuity
        """
        effective_rate = rate / compounding
        effective_periods = periods * compounding

        if type == 0:  # Ordinary annuity
            pv = pmt * (1 - 1 / (1 + effective_rate) ** effective_periods) / effective_rate
        else:  # Annuity due
            pv = pmt * (1 - 1 / (1 + effective_rate) ** effective_periods) / effective_rate * (1 + effective_rate)

        return pv

    def annuity_payment(self, pv: float, rate: float, periods: int, type: int = 0, compounding: int = 1) -> float:
        """
        Calculate required payment for a loan or annuity.

        Args:
            pv: Present value (loan amount)
            rate: Interest rate per period
            periods: Number of periods
            type: 0 for ordinary annuity, 1 for annuity due
            compounding: Compounding frequency per period

        Returns:
            float: Required payment per period
        """
        effective_rate = rate / compounding
        effective_periods = periods * compounding

        if type == 0:  # Ordinary annuity
            payment = pv * effective_rate / (1 - 1 / (1 + effective_rate) ** effective_periods)
        else:  # Annuity due
            payment = pv * effective_rate / (1 - 1 / (1 + effective_rate) ** effective_periods) / (1 + effective_rate)

        return payment

    def net_present_value(self, initial_investment: float, cash_flows: List[float], discount_rate: float) -> float:
        """
        Calculate Net Present Value of an investment.

        Args:
            initial_investment: Initial cash outflow
            cash_flows: List of expected cash flows
            discount_rate: Required rate of return

        Returns:
            float: Net Present Value
        """
        npv = -initial_investment

        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + discount_rate) ** (t + 1)

        return npv

    def internal_rate_of_return(
        self, initial_investment: float, cash_flows: List[float], guess: float = 0.1
    ) -> Optional[float]:
        """
        Calculate Internal Rate of Return using iterative method.

        Args:
            initial_investment: Initial cash outflow
            cash_flows: List of expected cash flows
            guess: Initial guess for IRR

        Returns:
            Optional[float]: Internal Rate of Return
        """

        def npv_function(rate):
            npv = -initial_investment
            for t, cash_flow in enumerate(cash_flows):
                npv += cash_flow / (1 + rate) ** (t + 1)
            return npv

        try:
            irr = newton(npv_function, guess)
            return irr if irr != guess else None
        except:
            return None

    def profitability_index(self, initial_investment: float, cash_flows: List[float], discount_rate: float) -> float:
        """
        Calculate Profitability Index.

        Args:
            initial_investment: Initial cash outflow
            cash_flows: List of expected cash flows
            discount_rate: Required rate of return

        Returns:
            float: Profitability Index
        """
        pv_future_cash_flows = sum(cf / (1 + discount_rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        return pv_future_cash_flows / initial_investment

    def loan_payment(
        self,
        principal: float,
        annual_rate: float,
        years: int,
        payments_per_year: int = 12,
    ) -> float:
        """
        Calculate loan payment amount.

        Args:
            principal: Loan principal
            annual_rate: Annual interest rate
            years: Loan term in years
            payments_per_year: Number of payments per year

        Returns:
            float: Payment amount per period
        """
        periodic_rate = annual_rate / payments_per_year
        total_periods = years * payments_per_year

        return (
            principal
            * (periodic_rate * (1 + periodic_rate) ** total_periods)
            / ((1 + periodic_rate) ** total_periods - 1)
        )

    def loan_amortization_schedule(
        self,
        principal: float,
        annual_rate: float,
        years: int,
        payments_per_year: int = 12,
    ) -> List[Dict[str, float]]:
        """
        Generate loan amortization schedule.

        Args:
            principal: Loan principal
            annual_rate: Annual interest rate
            years: Loan term in years
            payments_per_year: Number of payments per year

        Returns:
            List[Dict[str, float]]: Amortization schedule
        """
        payment = self.loan_payment(principal, annual_rate, years, payments_per_year)
        periodic_rate = annual_rate / payments_per_year
        total_periods = years * payments_per_year

        balance = principal
        schedule = []

        for period in range(1, total_periods + 1):
            interest_payment = balance * periodic_rate
            principal_payment = payment - interest_payment
            balance -= principal_payment

            schedule.append(
                {
                    "period": period,
                    "payment": payment,
                    "principal_payment": principal_payment,
                    "interest_payment": interest_payment,
                    "remaining_balance": max(0, balance),
                }
            )

        return schedule

    def effective_annual_rate(self, nominal_rate: float, compounding_freq: int) -> float:
        """
        Calculate effective annual rate.

        Args:
            nominal_rate: Nominal annual rate
            compounding_freq: Compounding frequency per year

        Returns:
            float: Effective annual rate
        """
        return (1 + nominal_rate / compounding_freq) ** compounding_freq - 1

    def continuous_compounding(self, pv: float, rate: float, time: float) -> float:
        """
        Calculate future value with continuous compounding.

        Args:
            pv: Present value
            rate: Annual interest rate
            time: Time in years

        Returns:
            float: Future value
        """
        return pv * math.exp(rate * time)

    def bond_price(
        self,
        face_value: float,
        coupon_rate: float,
        years: int,
        market_rate: float,
        payments_per_year: int = 2,
    ) -> float:
        """
        Calculate bond price.

        Args:
            face_value: Bond face value
            coupon_rate: Annual coupon rate
            years: Years to maturity
            market_rate: Market interest rate
            payments_per_year: Coupon payments per year

        Returns:
            float: Bond price
        """
        coupon_payment = face_value * coupon_rate / payments_per_year
        periodic_market_rate = market_rate / payments_per_year
        total_periods = years * payments_per_year

        # Present value of coupon payments
        pv_coupons = 0
        for t in range(1, total_periods + 1):
            pv_coupons += coupon_payment / (1 + periodic_market_rate) ** t

        # Present value of face value
        pv_face = face_value / (1 + periodic_market_rate) ** total_periods

        return pv_coupons + pv_face

    def bond_yield_to_maturity(
        self,
        bond_price: float,
        face_value: float,
        coupon_rate: float,
        years: int,
        payments_per_year: int = 2,
        guess: float = 0.05,
    ) -> Optional[float]:
        """
        Calculate bond yield to maturity.

        Args:
            bond_price: Current bond price
            face_value: Bond face value
            coupon_rate: Annual coupon rate
            years: Years to maturity
            payments_per_year: Coupon payments per year
            guess: Initial guess for YTM

        Returns:
            Optional[float]: Yield to maturity
        """

        def price_function(yield_rate):
            return self.bond_price(face_value, coupon_rate, years, yield_rate, payments_per_year) - bond_price

        try:
            ytm = newton(price_function, guess)
            return ytm
        except:
            return None

    def retirement_planning(
        self,
        current_age: int,
        retirement_age: int,
        current_savings: float,
        annual_contribution: float,
        expected_return: float,
        desired_income: float,
        life_expectancy: int,
        inflation_rate: float = 0.03,
    ) -> Dict[str, float]:
        """
        Comprehensive retirement planning analysis.

        Args:
            current_age: Current age
            retirement_age: Planned retirement age
            current_savings: Current retirement savings
            annual_contribution: Annual contribution amount
            expected_return: Expected annual return
            desired_income: Desired annual retirement income
            life_expectancy: Life expectancy
            inflation_rate: Annual inflation rate

        Returns:
            Dict[str, float]: Retirement analysis results
        """
        years_to_retirement = retirement_age - current_age
        years_in_retirement = life_expectancy - retirement_age

        # Future value of current savings
        fv_savings = self.future_value_single(current_savings, expected_return, years_to_retirement)

        # Future value of contributions
        if annual_contribution > 0:
            fv_contributions = self.future_value_annuity(annual_contribution, expected_return, years_to_retirement)
        else:
            fv_contributions = 0

        total_at_retirement = fv_savings + fv_contributions

        # Present value of retirement income need
        retirement_income_pv = self.present_value_annuity(desired_income, expected_return, years_in_retirement)

        # Inflation adjustment
        inflation_adjusted_need = retirement_income_pv * (1 + inflation_rate) ** years_to_retirement

        # Savings gap
        savings_gap = max(0, inflation_adjusted_need - total_at_retirement)

        # Required additional contribution
        if years_to_retirement > 0 and expected_return > 0:
            required_contribution = self.annuity_payment(savings_gap, expected_return, years_to_retirement)
        else:
            required_contribution = 0

        return {
            "future_value_savings": fv_savings,
            "future_value_contributions": fv_contributions,
            "total_at_retirement": total_at_retirement,
            "retirement_income_needed": retirement_income_pv,
            "inflation_adjusted_need": inflation_adjusted_need,
            "savings_gap": savings_gap,
            "required_annual_contribution": required_contribution,
            "years_to_retirement": years_to_retirement,
            "years_in_retirement": years_in_retirement,
        }

    def sensitivity_analysis(
        self, base_scenario: Dict[str, float], variable: str, values: List[float]
    ) -> List[Dict[str, float]]:
        """
        Perform sensitivity analysis on TVM calculations.

        Args:
            base_scenario: Base scenario parameters
            variable: Variable to test (e.g., 'expected_return', 'years_to_retirement')
            values: List of values to test

        Returns:
            List[Dict[str, float]]: Sensitivity analysis results
        """
        results = []

        for value in values:
            scenario = base_scenario.copy()
            scenario[variable] = value

            if "analysis_type" in scenario and scenario["analysis_type"] == "retirement":
                result = self.retirement_planning(
                    scenario["current_age"],
                    scenario["retirement_age"],
                    scenario["current_savings"],
                    scenario["annual_contribution"],
                    scenario["expected_return"],
                    scenario["desired_income"],
                    scenario["life_expectancy"],
                    scenario.get("inflation_rate", 0.03),
                )
            else:
                # Generic investment calculation
                result = {
                    "future_value": self.future_value_single(
                        scenario.get("principal", 0),
                        scenario.get("rate", 0),
                        scenario.get("periods", 0),
                    ),
                    "test_value": value,
                }

            results.append(result)

        return results

    def create_comparison_table(self, scenarios: List[Dict[str, str]]) -> str:
        """
        Create a formatted comparison table for different scenarios.

        Args:
            scenarios: List of scenario dictionaries

        Returns:
            str: Formatted comparison table
        """
        table = "\n" + "=" * 80 + "\n"
        table += f"{'Scenario':<20} {'Future Value':<15} {'Present Value':<15} {'Payment':<12}\n"
        table += "=" * 80 + "\n"

        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            fv = scenario.get("future_value", 0)
            pv = scenario.get("present_value", 0)
            pmt = scenario.get("payment", 0)

            table += f"{name:<20} ${fv:<13.2f} ${pv:<13.2f} ${pmt:<10.2f}\n"

        table += "=" * 80 + "\n"
        return table


def main():
    """
    Main function demonstrating TVM calculations.
    """
    print("=== Time Value of Money Calculations ===\n")

    # Initialize the class
    tvm = TimeValueMoney()

    # Demo 1: Basic Investment Analysis
    print("1. Investment Analysis:")
    principal = 10000
    rate = 0.08
    years = 10

    fv = tvm.future_value_single(principal, rate, years)
    pv = tvm.present_value_single(fv, rate, years)

    print(f"Initial investment: ${principal:,.2f}")
    print(f"Future value after {years} years at {rate:.1%}: ${fv:,.2f}")
    print(f"Present value: ${pv:,.2f}\n")

    # Demo 2: Loan Analysis
    print("2. Loan Analysis:")
    loan_amount = 300000
    loan_rate = 0.045
    loan_term = 30

    monthly_payment = tvm.loan_payment(loan_amount, loan_rate, loan_term)
    print(f"Loan amount: ${loan_amount:,.2f}")
    print(f"Interest rate: {loan_rate:.1%}")
    print(f"Term: {loan_term} years")
    print(f"Monthly payment: ${monthly_payment:.2f}")

    # Show first few payments
    schedule = tvm.loan_amortization_schedule(loan_amount, loan_rate, loan_term)
    print("\nFirst payment breakdown:")
    print(f"Principal: ${schedule[0]['principal_payment']:.2f}")
    print(f"Interest: ${schedule[0]['interest_payment']:.2f}")
    print(f"Remaining balance: ${schedule[0]['remaining_balance']:.2f}\n")

    # Demo 3: Retirement Planning
    print("3. Retirement Planning:")
    retirement_analysis = tvm.retirement_planning(
        current_age=30,
        retirement_age=65,
        current_savings=25000,
        annual_contribution=5000,
        expected_return=0.07,
        desired_income=60000,
        life_expectancy=90,
    )

    print(f"Total at retirement: ${retirement_analysis['total_at_retirement']:,.2f}")
    print(f"Savings gap: ${retirement_analysis['savings_gap']:,.2f}")
    print(f"Required additional contribution: ${retirement_analysis['required_annual_contribution']:,.2f}\n")

    # Demo 4: Bond Pricing
    print("4. Bond Pricing:")
    bond_price = tvm.bond_price(face_value=1000, coupon_rate=0.05, years=10, market_rate=0.06)

    ytm = tvm.bond_yield_to_maturity(bond_price=950, face_value=1000, coupon_rate=0.05, years=10)

    print(f"Bond price: ${bond_price:.2f}")
    print(f"Yield to maturity: {ytm:.4f}" if ytm else "Could not calculate YTM")

    # Demo 5: NPV and IRR
    print("\n5. Capital Budgeting:")
    initial_investment = 100000
    cash_flows = [30000, 35000, 40000, 45000, 50000]

    npv = tvm.net_present_value(initial_investment, cash_flows, 0.12)
    irr = tvm.internal_rate_of_return(initial_investment, cash_flows)

    print(f"Initial investment: ${initial_investment:,.2f}")
    print(f"NPV at 12% discount rate: ${npv:,.2f}")
    print(f"IRR: {irr:.4f}" if irr else "Could not calculate IRR")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
