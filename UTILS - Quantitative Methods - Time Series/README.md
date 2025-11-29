# Quantitative Methods – Time Series Utility

## 📋 Overview

This utility introduces core time-series techniques used in quantitative finance. It serves as a bridge between the intermediate and advanced curriculum (`Documentation/Programs/level3_financial.py` and `level4_advanced.py`) and gives you reusable helpers for analyzing historical price data.

## 🎯 Key Skills
- Generating and cleaning time-series price data
- Calculating rolling statistics (moving averages, volatility)
- Computing autocorrelation and partial autocorrelation
- Performing stationarity checks (Augmented Dickey-Fuller)
- Building a simple AR(1) forecast as a baseline model

## 📂 Files
- `time_series_tools.py`: Guided walkthrough with annotated prints and helper functions

## 🚀 How to Run
```bash
python time_series_tools.py
```
Open the script while it runs to follow the inline commentary.

## 📚 Dependencies
- pandas
- numpy
- statsmodels (for the ADF test)

Install them with:
```bash
pip install pandas numpy statsmodels
```

## 🧠 Practice Ideas
- Swap the simulated data with real prices from `yfinance`
- Experiment with different rolling window lengths for volatility
- Extend the AR(1) section to ARIMA using `statsmodels.tsa.arima.model`

## 🔗 Related Modules
- `Documentation/Programs/level3_financial.py`
- `Documentation/Programs/level4_advanced.py`
- `UTILS - Technical Indicators/`
