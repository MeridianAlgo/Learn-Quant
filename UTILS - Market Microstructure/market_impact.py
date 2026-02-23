"""
Market Impact Models

Implementation of various market impact models for estimating the cost
of trading and optimizing execution strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Trade:
    """Represents a single trade."""
    quantity: float
    price: float
    side: str  # 'buy' or 'sell'
    timestamp: Optional[float] = None


class MarketImpactModel(ABC):
    """Abstract base class for market impact models."""

    @abstractmethod
    def calculate_impact(self, trade: Trade, market_data: Dict[str, Any]) -> float:
        """Calculate the market impact of a trade."""
        pass


class AlmgrenChrissModel(MarketImpactModel):
    """
    Almgren-Chriss market impact model.

    Models both temporary and permanent market impact based on trade size
    and market volatility.
    """

    def __init__(self, eta: float = 0.001, gamma: float = 0.01,
                 lambda_temp: float = 0.0001):
        """
        Initialize Almgren-Chriss model.

        Args:
            eta: Permanent impact coefficient
            gamma: Temporary impact coefficient
            lambda_temp: Temporary impact scaling factor
        """
        self.eta = eta
        self.gamma = gamma
        self.lambda_temp = lambda_temp

    def calculate_impact(self, trade: Trade, market_data: Dict[str, Any]) -> float:
        """
        Calculate market impact using Almgren-Chriss model.

        Args:
            trade: Trade object
            market_data: Dictionary containing market data (volume, volatility, etc.)

        Returns:
            Total market impact cost
        """
        volume = market_data.get('daily_volume', 1e6)
        volatility = market_data.get('volatility', 0.02)
        mid_price = market_data.get('mid_price', trade.price)

        # Trade size as fraction of daily volume
        trade_fraction = abs(trade.quantity) / volume

        # Permanent impact (linear in trade size)
        permanent_impact = self.eta * trade_fraction * mid_price

        # Temporary impact (square-root law)
        temporary_impact = self.gamma * np.sqrt(trade_fraction) * mid_price

        # Adjust for volatility
        volatility_adjustment = volatility * np.sqrt(self.lambda_temp)
        temporary_impact *= (1 + volatility_adjustment)

        return permanent_impact + temporary_impact


class SquareRootModel(MarketImpactModel):
    """
    Square-root market impact model.

    Based on the square-root law of market impact.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.5):
        """
        Initialize square-root model.

        Args:
            alpha: Impact coefficient
            beta: Impact exponent (typically 0.5 for square-root law)
        """
        self.alpha = alpha
        self.beta = beta

    def calculate_impact(self, trade: Trade, market_data: Dict[str, Any]) -> float:
        """Calculate market impact using square-root model."""
        volume = market_data.get('daily_volume', 1e6)
        mid_price = market_data.get('mid_price', trade.price)

        # Relative trade size
        relative_size = abs(trade.quantity) / volume

        # Square-root impact
        impact = self.alpha * (relative_size ** self.beta) * mid_price

        return impact


class LinearModel(MarketImpactModel):
    """
    Linear market impact model.

    Simple linear model for market impact estimation.
    """

    def __init__(self, impact_coefficient: float = 0.0001):
        """
        Initialize linear model.

        Args:
            impact_coefficient: Linear impact coefficient
        """
        self.impact_coefficient = impact_coefficient

    def calculate_impact(self, trade: Trade, market_data: Dict[str, Any]) -> float:
        """Calculate market impact using linear model."""
        volume = market_data.get('daily_volume', 1e6)
        mid_price = market_data.get('mid_price', trade.price)

        # Linear impact based on trade fraction
        trade_fraction = abs(trade.quantity) / volume
        impact = self.impact_coefficient * trade_fraction * mid_price

        return impact


class MarketImpactCalculator:
    """
    Comprehensive market impact calculator with multiple models.
    """

    def __init__(self):
        self.models = {
            'almgren_chriss': AlmgrenChrissModel(),
            'square_root': SquareRootModel(),
            'linear': LinearModel()
        }

    def add_custom_model(self, name: str, model: MarketImpactModel) -> None:
        """Add a custom market impact model."""
        self.models[name] = model

    def calculate_impact(self, trade: Trade, market_data: Dict[str, Any],
                        model_name: str = 'almgren_chriss') -> float:
        """
        Calculate market impact using specified model.

        Args:
            trade: Trade object
            market_data: Market data dictionary
            model_name: Name of the model to use

        Returns:
            Market impact cost
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

        return self.models[model_name].calculate_impact(trade, market_data)

    def compare_models(self, trade: Trade, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare impact across all available models.

        Args:
            trade: Trade object
            market_data: Market data dictionary

        Returns:
            Dictionary of model names and their impact calculations
        """
        results = {}
        for name, model in self.models.items():
            results[name] = model.calculate_impact(trade, market_data)

        return results

    def optimize_execution_size(self, target_quantity: float, market_data: Dict[str, Any],
                              model_name: str = 'almgren_chriss',
                              max_slices: int = 10) -> Tuple[float, float]:
        """
        Optimize execution size to minimize market impact.

        Args:
            target_quantity: Total quantity to execute
            market_data: Market data dictionary
            model_name: Model to use for optimization
            max_slices: Maximum number of slices to consider

        Returns:
            Tuple of (optimal_slice_size, minimum_impact)
        """
        best_impact = float('inf')
        best_slice_size = target_quantity

        # Try different slice sizes
        for num_slices in range(1, max_slices + 1):
            slice_size = target_quantity / num_slices

            # Calculate total impact for this slicing strategy
            total_impact = 0
            for _ in range(num_slices):
                slice_trade = Trade(slice_size, market_data.get('mid_price', 100), 'buy')
                impact = self.calculate_impact(slice_trade, market_data, model_name)
                total_impact += impact

            if total_impact < best_impact:
                best_impact = total_impact
                best_slice_size = slice_size

        return best_slice_size, best_impact

    def calculate_implementation_shortfall(self, trades: list,
                                         market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate implementation shortfall for a series of trades.

        Args:
            trades: List of Trade objects
            market_data: Market data dictionary

        Returns:
            Dictionary with shortfall components
        """
        if not trades:
            return {'total_shortfall': 0, 'market_impact': 0, 'timing_cost': 0}

        # Initial decision price
        decision_price = market_data.get('decision_price', trades[0].price)

        # Calculate execution prices
        execution_prices = [trade.price for trade in trades]
        total_quantity = sum(trade.quantity for trade in trades)

        # Weighted average execution price
        avg_execution_price = sum(p * q for p, q in zip(execution_prices,
                                                       [t.quantity for t in trades])) / total_quantity

        # Market impact cost
        market_impact = sum(self.calculate_impact(trade, market_data) for trade in trades)

        # Timing cost (price movement from decision to execution)
        timing_cost = (avg_execution_price - decision_price) * total_quantity

        # Total implementation shortfall
        total_shortfall = market_impact + timing_cost

        return {
            'total_shortfall': total_shortfall,
            'market_impact': market_impact,
            'timing_cost': timing_cost,
            'avg_execution_price': avg_execution_price,
            'decision_price': decision_price
        }


def main():
    """Example usage of market impact models."""
    # Create calculator
    calculator = MarketImpactCalculator()

    # Example trade
    trade = Trade(quantity=10000, price=100.0, side='buy')

    # Example market data
    market_data = {
        'daily_volume': 1000000,
        'volatility': 0.02,
        'mid_price': 100.0,
        'decision_price': 99.5
    }

    print("Market Impact Analysis")
    print("=" * 50)

    # Compare models
    impacts = calculator.compare_models(trade, market_data)
    print(f"Trade: {trade.quantity} shares at ${trade.price}")
    print(f"Daily Volume: {market_data['daily_volume']:,} shares")
    print("\nImpact by Model:")
    for model, impact in impacts.items():
        print(f"  {model}: ${impact:.4f}")

    # Optimize execution
    optimal_slice, min_impact = calculator.optimize_execution_size(
        trade.quantity, market_data, 'almgren_chriss'
    )
    print("\nOptimal Execution:")
    print(f"  Slice size: {optimal_slice:.0f} shares")
    print(f"  Minimum impact: ${min_impact:.4f}")

    # Implementation shortfall for multiple trades
    trades = [
        Trade(5000, 100.1, 'buy'),
        Trade(5000, 100.2, 'buy')
    ]
    shortfall = calculator.calculate_implementation_shortfall(trades, market_data)
    print("\nImplementation Shortfall:")
    print(f"  Total: ${shortfall['total_shortfall']:.4f}")
    print(f"  Market Impact: ${shortfall['market_impact']:.4f}")
    print(f"  Timing Cost: ${shortfall['timing_cost']:.4f}")


if __name__ == "__main__":
    main()
