# Advanced Python – Object-Oriented Programming

## 📋 Overview

Object-Oriented Programming (OOP) is essential for building scalable, maintainable trading systems and financial applications. Learn to organize code using classes, objects, and OOP principles.

## 🎯 Key Concepts

### **Classes and Objects**
- **Class**: Blueprint for creating objects
- **Object**: Instance of a class with specific data
- **Attributes**: Data stored in objects
- **Methods**: Functions that operate on objects

### **Encapsulation**
- **Private attributes**: Use `_` or `__` prefix
- **Properties**: Control attribute access with `@property`
- **Getters/Setters**: Validate data before modification
- **Information hiding**: Hide implementation details

### **Inheritance**
- **Parent/Child classes**: Reuse and extend functionality
- **Method overriding**: Customize inherited behavior
- **`super()`**: Call parent class methods
- **Multiple inheritance**: Inherit from multiple parents

### **Polymorphism**
- **Method overloading**: Same method, different signatures
- **Duck typing**: "If it walks like a duck..."
- **Abstract classes**: Define interfaces
- **Type flexibility**: Work with different object types

### **Special Methods**
- **`__init__`**: Constructor
- **`__str__`**: User-friendly string representation
- **`__repr__`**: Developer-friendly representation
- **`__eq__`, `__lt__`, etc.**: Comparison operators

## 💻 Key Examples

### Basic Class
```python
class Stock:
    def __init__(self, ticker: str, price: float, shares: int = 0):
        self.ticker = ticker
        self.price = price
        self.shares = shares
    
    def get_value(self) -> float:
        return self.price * self.shares
    
    def __str__(self) -> str:
        return f"{self.ticker}: ${self.price:.2f} × {self.shares}"

# Create objects
aapl = Stock("AAPL", 175.50, 50)
print(aapl.get_value())  # $8,775.00
```

### Properties
```python
class Trade:
    def __init__(self, entry: float, exit: float, shares: int):
        self.entry = entry
        self.exit = exit
        self.shares = shares
    
    @property
    def pnl(self) -> float:
        """Calculate P&L automatically."""
        return (self.exit - self.entry) * self.shares

trade = Trade(100, 105, 50)
print(trade.pnl)  # $250.00 (no parentheses needed!)
```

### Composition
```python
class Portfolio:
    def __init__(self, name: str):
        self.name = name
        self.holdings: Dict[str, Stock] = {}
    
    def add_stock(self, stock: Stock):
        self.holdings[stock.ticker] = stock
    
    def get_total_value(self) -> float:
        return sum(s.get_value() for s in self.holdings.values())
```

## 📂 Files
- `oop_tutorial.py`: Comprehensive OOP tutorial with trading classes

## 🚀 How to Run
```bash
python oop_tutorial.py
```

## 🧠 Financial Applications

### 1. Trading Systems
```python
class TradingStrategy:
    def generate_signal(self, data): ...
    def calculate_position_size(self): ...
    def execute_trade(self): ...

class MomentumStrategy(TradingStrategy):
    # Inherit and customize
    pass

class MeanReversionStrategy(TradingStrategy):
    # Different implementation
    pass
```

### 2. Portfolio Management
```python
class Asset:
    # Stocks, Bonds, Options, etc.
    pass

class Portfolio:
    def __init__(self):
        self.assets: List[Asset] = []
    
    def rebalance(self): ...
    def calculate_risk(self): ...
```

### 3. Order Management
```python
class Order:
    # Base order class
    pass

class MarketOrder(Order):
    pass

class LimitOrder(Order):
    def __init__(self, limit_price: float):
        self.limit_price = limit_price

class StopOrder(Order):
    def __init__(self, stop_price: float):
        self.stop_price = stop_price
```

### 4. Risk Management
```python
class RiskManager:
    def check_position_limit(self, position): ...
    def calculate_var(self, portfolio): ...
    def enforce_stop_loss(self, trade): ...
```

## 💡 Best Practices

### Single Responsibility
```python
✓ DO:
class PriceDataFetcher:
    def fetch_prices(self): ...

class ReturnCalculator:
    def calculate_returns(self, prices): ...

✗ DON'T:
class DataManager:
    def fetch_prices(self): ...
    def calculate_returns(self): ...
    def plot_charts(self): ...
    def send_email(self): ...
```

### Composition Over Inheritance
```python
✓ DO:
class Portfolio:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.rebalancer = Rebalancer()

✗ DON'T:
class Portfolio(RiskManager, Rebalancer, Reporter, Optimizer):
    # Too many parent classes!
    pass
```

### Use Type Hints
```python
✓ DO:
class Trade:
    def __init__(self, ticker: str, price: float, shares: int):
        self.ticker = ticker
        self.price = price
        self.shares = shares

✗ DON'T:
class Trade:
    def __init__(self, ticker, price, shares):  # No type info
        ...
```

## 🎓 Practice Problems

1. **Stock Portfolio Tracker**
   - Create `Stock` and `Portfolio` classes
   - Track purchases, sales, and current value
   - Calculate allocation percentages

2. **Order Book**
   - Create `Order` base class
   - Implement `BuyOrder` and `SellOrder` subclasses
   - Track order status (pending, filled, cancelled)

3. **Backtesting Framework**
   - Create `Strategy` base class
   - Implement specific strategies as subclasses
   - Track trades and performance

4. **Option Pricing**
   - Create `Option` base class
   - Implement `CallOption` and `PutOption`
   - Calculate Greeks (delta, gamma, etc.)

## 📖 Design Patterns

### Factory Pattern
```python
class OrderFactory:
    @staticmethod
    def create_order(order_type: str, **kwargs) -> Order:
        if order_type == "MARKET":
            return MarketOrder(**kwargs)
        elif order_type == "LIMIT":
            return LimitOrder(**kwargs)
        # ...
```

### Observer Pattern
```python
class PriceObserver:
    def update(self, price): ...

class PriceMonitor:
    def __init__(self):
        self.observers: List[PriceObserver] = []
    
    def attach(self, observer): ...
    def notify(self, price): ...
```

### Strategy Pattern
```python
class TradingStrategy(ABC):
    @abstractmethod
    def execute(self, data): ...

class Bot:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
    
    def run(self):
        self.strategy.execute(data)
```

---

*Master OOP to build professional-grade trading systems and financial applications!*
