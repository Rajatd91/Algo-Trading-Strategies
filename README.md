# Algorithmic Trading Coursework (COMP0051)

This repository contains the complete algorithmic trading pipeline developed for the COMP0051 Coursework (2025/26). It implements and backtests two distinct trading strategies on high-frequency (hourly) cryptocurrency data.

## Project Structure

* **`1_data_download.py`**: Downloads and cleans hourly HLOCV data for five cryptocurrencies (BTC, ETH, DOGE, BNB, XRP) from Binance (Jan 2024 to Mar 2026). Generates excess returns against the FRED Federal Funds Rate and handles 5-$\sigma$ outlier clipping.
* **`2_strategy.py`**: Implements the core trading strategies:
  * **Strategy 1 (Mean-Reversion)**: Uses a 168h rolling z-score signal and allocates position sizes using Markowitz Mean-Variance Optimisation (MVO) with Ledoit-Wolf shrinkage.
  * **Strategy 2 (Trend-Following)**: Uses a 12h/96h Fast-Slow Moving Average crossover with time-series momentum volatility scaling.
* **`3_4_costs_performance.py`**: Estimates realistic microstructure slippage using the Roll (1984) model (floored at Binance's execution fees) and computes net out-of-sample performance metrics (Sharpe, Sortino, Calmar, Turnover, Peak Drawdown).
* **`5_next_steps.py`**: Conducts advanced performance testing, including rolling time-horizon Sharpe dynamics, bullish/bearish market regime sensitivity, executing risk feasibility limits, and mathematical 50/50 combination metrics.
* **`rebal_comparison.py`**: A parallel evaluation test suite mapping turnover friction across 6h, 24h, 168h, 336h, and 720h (monthly) rebalancing intervals.
* **`generate_report.py`**: Automatically constructs the final 5-page PDF report leveraging `fpdf`.

## Core Constraints Implemented
- Strict $V_0$ Initial Capital: **$10,000**
- Hard Gross Exposure Limit (10x Leverage): **$100,000 Cap**
- Both strategies successfully pass these constraints dynamically at every execution timestep.

## Dependencies
- pandas
- numpy
- matplotlib
- scipy
- fpdf
- statsmodels

## Usage
Run the scripts sequentially (`1` through `5`) to reproduce the data cleaning, strategy array formations, and performance charts generated in the `data/` folder.
