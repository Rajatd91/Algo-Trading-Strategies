"""
COMP0051 - Algorithmic Trading Coursework
Task 2: Trading Strategies

Strategy 1: Multi-Asset Mean-Reversion with MVO
    - Reference: Avellaneda & Lee (2010), Markowitz (1952), Meucci (2009),
                 Cartea et al. (2015)

Strategy 2: Trend-Following (Time-Series Momentum)
    - Reference: Moskowitz, Ooi & Pedersen (2012), Hurst, Ooi & Pedersen (2017),
                 Jegadeesh & Titman (1993), Cartea et al. (2015)

Capital rules:
    - Each strategy starts with V0 = $10,000 USDT
    - Gross exposure constraint: sum(|theta_i|) <= $100,000 (max 10x leverage)
    - Leverage is DYNAMIC: positions scale with signal strength
    - When signals are weak → stay close to $10k (1x leverage, no borrowing)
    - When signals are strong → use more leverage (up to 10x max)
    - $100,000 is a HARD CAP, not a target
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ─── Configuration ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COINS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT']
COIN_LABELS = ['BTC', 'ETH', 'DOGE', 'BNB', 'XRP']

V0 = 10_000                   # Starting capital in USDT
MAX_GROSS_EXPOSURE = 100_000   # Hard cap: max 10x leverage
TRAIN_END = pd.Timestamp('2024-12-31 23:00:00')


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load cleaned price and return data for all coins."""
    prices = pd.DataFrame()
    returns = pd.DataFrame()

    for coin in COINS:
        df_clean = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet"))
        df_ret = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_returns.parquet"))
        prices[coin] = df_clean['close']
        returns[coin] = df_ret['excess_return']

    common_idx = prices.index.intersection(returns.index)
    prices = prices.loc[common_idx]
    returns = returns.loc[common_idx]
    return prices, returns


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: MULTI-ASSET MEAN-REVERSION WITH MVO
# ═══════════════════════════════════════════════════════════════════════

def generate_mr_signals(prices, lookback_slow):
    """
    Mean-reversion alpha signals following Avellaneda & Lee (2010).

    For each coin at each hour:
    1. Compute the moving average of the last 'lookback_slow' hours
    2. Compare current price to that average
    3. Normalise by volatility so all coins are comparable

    Result: a number for each coin at each hour
    - Positive = price is below average = BUY signal
    - Negative = price is above average = SELL signal
    - Bigger number = stronger signal
    """
    close = prices[COINS].values
    n, k = close.shape
    signals = np.full((n, k), np.nan)

    for t in range(lookback_slow, n):
        for i in range(k):
            window = close[t - lookback_slow:t + 1, i]
            ma = window[:-1].mean()  # Average of previous lookback hours
            rets = np.diff(window[:-1]) / window[:-2]
            vol = rets.std() if len(rets) > 1 else 1e-8

            if vol > 0 and close[t, i] > 0:
                signals[t, i] = (ma - close[t, i]) / (close[t, i] * vol)

    # Cross-sectional z-score: rank signals across the 5 coins
    # This makes them comparable (BTC signal vs DOGE signal)
    for t in range(n):
        row = signals[t]
        if np.all(np.isnan(row)):
            continue
        mu = np.nanmean(row)
        sigma = np.nanstd(row)
        if sigma > 0:
            signals[t] = (row - mu) / sigma

    return pd.DataFrame(signals, index=prices.index, columns=COINS)


def mvo_optimize(mu, sigma, theta_prev, gamma, tc_penalty, max_gross):
    """
    Mean-Variance Optimization with transaction cost penalty.

    This solves:
        max  mu'θ  -  (γ/2)θ'Σθ  -  λ·||θ - θ_prev||₁
        s.t. Σ|θ_i| ≤ max_gross

    In plain English:
    - mu'θ = "I want high expected returns"
    - (γ/2)θ'Σθ = "but I don't want too much risk" (γ controls this)
    - λ·||θ - θ_prev|| = "and I don't want to trade too much (costly)"
    - Σ|θ_i| ≤ max_gross = "and I can't exceed $100k total"

    When γ is LARGE: the optimizer is very risk-averse → small positions
    When γ is SMALL: the optimizer takes big positions

    Reference: Markowitz (1952), Cartea et al. (2015) Ch.7
    """
    n_assets = len(mu)

    def neg_obj(theta):
        ret = mu @ theta
        risk = 0.5 * gamma * theta @ sigma @ theta
        cost = tc_penalty * np.sum(np.abs(theta - theta_prev))
        return -(ret - risk - cost)

    constraints = [{'type': 'ineq', 'fun': lambda t: max_gross - np.sum(np.abs(t))}]
    bounds = [(-max_gross, max_gross)] * n_assets

    result = minimize(neg_obj, x0=theta_prev, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 100, 'ftol': 1e-8})

    if result.success:
        return result.x

    # Fallback: closed-form + scaling
    try:
        theta_raw = (1.0 / gamma) * np.linalg.solve(sigma, mu)
    except np.linalg.LinAlgError:
        theta_raw = mu / gamma

    blend = tc_penalty / (tc_penalty + 0.01)
    theta_out = (1 - blend) * theta_raw + blend * theta_prev
    gross = np.sum(np.abs(theta_out))
    if gross > max_gross:
        theta_out *= max_gross / gross
    return theta_out


def backtest_strategy1(prices, returns, params, train_end, rebal_freq=720):
    """
    Backtest mean-reversion with MVO.

    TWO-STEP position sizing:
    Step 1: MVO decides the RELATIVE weights across 5 coins
    Step 2: Signal confidence decides the TOTAL EXPOSURE

    rebal_freq: hours between rebalances
        - 6 for calibration (more data points for Sharpe estimation)
        - 720 (monthly) for production (lower costs)
        Reference: Moskowitz, Ooi & Pedersen (2012) use monthly rebalancing
    """
    lookback_slow = params['lookback_slow']
    lookback_cov = params['lookback_cov']
    gamma = params['gamma']
    kappa = params['kappa']
    tc_penalty = params['tc_penalty']
    confidence_threshold = params.get('confidence_threshold', 2.0)

    n = len(prices)
    signals = generate_mr_signals(prices, lookback_slow)
    signals_vals = signals.values
    returns_vals = returns[COINS].values

    theta_arr = np.zeros((n, len(COINS)))
    theta_prev = np.zeros(len(COINS))

    start_idx = max(lookback_slow, lookback_cov) + 1
    REBAL_FREQ = rebal_freq
    last_sigma = np.eye(len(COINS)) * 1e-4

    for t in range(start_idx, n):
        if t % REBAL_FREQ != 0:
            # NOT rebalancing → let positions drift with the market
            # If BTC went up 1%, our BTC position is now worth 1% more
            # We're just HOLDING, not trading → no transaction costs
            if t > 0:
                theta_arr[t] = theta_arr[t - 1] * (1 + returns_vals[t])
            else:
                theta_arr[t] = theta_prev
            # Hard cap: if drift pushes gross exposure above $100k, scale down
            # This is a risk limit — we'd partially liquidate to stay within bounds
            gross_drift = np.sum(np.abs(theta_arr[t]))
            if gross_drift > MAX_GROSS_EXPOSURE:
                theta_arr[t] *= MAX_GROSS_EXPOSURE / gross_drift
            continue

        s_t = signals_vals[t]

        if np.all(np.isnan(s_t)) or np.all(s_t == 0):
            # No signal → reduce position toward zero (fade out)
            theta_arr[t] = theta_prev * 0.9
            theta_prev = theta_arr[t].copy()
            continue

        s_t = np.nan_to_num(s_t, 0.0)
        mu_t = kappa * s_t

        # Update covariance
        ret_window = returns_vals[max(0, t - lookback_cov):t]
        if len(ret_window) >= 20:
            cov_raw = np.cov(ret_window, rowvar=False)
            if not (np.any(np.isnan(cov_raw)) or np.any(np.isinf(cov_raw))):
                diag = np.diag(np.diag(cov_raw))
                last_sigma = 0.5 * cov_raw + 0.5 * diag

        # STEP 1: MVO gives us raw positions (relative weights)
        theta_raw = mvo_optimize(mu_t, last_sigma, theta_prev, gamma,
                                  tc_penalty, MAX_GROSS_EXPOSURE)

        # STEP 2: Scale total exposure by signal confidence
        #
        # signal_confidence = how strong are the signals right now?
        # avg|signal| = 0 → no opportunities → stay at base capital ($10k)
        # avg|signal| = 1 → moderate opportunities → use some leverage
        # avg|signal| ≥ threshold → strong opportunities → use max leverage
        #
        # target_exposure = V0 + (MAX - V0) × min(1, confidence / threshold)
        #
        # Example with threshold = 2.0:
        #   confidence = 0.0 → target = $10k (1x)
        #   confidence = 0.5 → target = $32.5k (3.25x)
        #   confidence = 1.0 → target = $55k (5.5x)
        #   confidence = 2.0 → target = $100k (10x)

        signal_confidence = np.mean(np.abs(s_t))
        leverage_fraction = min(1.0, signal_confidence / confidence_threshold)
        target_exposure = V0 + (MAX_GROSS_EXPOSURE - V0) * leverage_fraction

        # Scale MVO output to match target exposure
        gross_raw = np.sum(np.abs(theta_raw))
        if gross_raw > 0:
            theta_scaled = theta_raw * (target_exposure / gross_raw)
        else:
            theta_scaled = theta_raw

        # Hard cap: never exceed $100k
        gross_final = np.sum(np.abs(theta_scaled))
        if gross_final > MAX_GROSS_EXPOSURE:
            theta_scaled *= MAX_GROSS_EXPOSURE / gross_final

        theta_arr[t] = theta_scaled
        theta_prev = theta_scaled.copy()

    theta = pd.DataFrame(theta_arr, index=prices.index, columns=COINS)
    return theta, signals


def calibrate_strategy1(prices_train, returns_train):
    """
    Grid search to find best parameters for Strategy 1.

    We test different combinations of:
    - lookback_slow: how many hours of history for the MA signal
    - gamma: how risk-averse the MVO is (HIGHER = smaller positions)
    - kappa: how strongly to scale the signal into expected return

    For each combination, we backtest on the training data and compute
    the Sharpe ratio. We pick the combination with the highest Sharpe.
    """
    print("\n  Calibrating Strategy 1 parameters (grid search)...")
    print("  (This takes several minutes)")

    best_sharpe = -np.inf
    best_params = None
    results = []

    returns_vals = returns_train[COINS].values

    # gamma controls relative allocation between coins (MVO step)
    # kappa scales signals into expected returns
    # confidence_threshold controls how much leverage to use:
    #   threshold=1.0 → reaches 10x leverage more easily
    #   threshold=2.0 → needs stronger signals for high leverage
    #   threshold=3.0 → very conservative, rarely uses high leverage
    for lookback_slow in [120, 168, 336]:
        for gamma in [1e-4, 5e-4, 1e-3]:
            for confidence_threshold in [1.0, 2.0, 3.0]:
                params = {
                    'lookback_slow': lookback_slow,
                    'lookback_cov': 168,
                    'gamma': gamma,
                    'kappa': 0.01,  # fixed — confidence_threshold controls sizing
                    'tc_penalty': 0.001,
                    'confidence_threshold': confidence_threshold,
                }

                theta, _ = backtest_strategy1(
                    prices_train, returns_train, params, TRAIN_END,
                    rebal_freq=6  # Use 6h for calibration (more data points)
                )

                theta_vals = theta[COINS].values
                start = max(lookback_slow, 168) + 2
                pnl = np.nansum(theta_vals[start:-1] * returns_vals[start + 1:], axis=1)

                if len(pnl) == 0 or np.std(pnl) == 0:
                    continue

                sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(24 * 365)

                # Track average leverage used
                gross = np.sum(np.abs(theta_vals[start:]), axis=1)
                avg_lev = np.mean(gross[gross > 0]) / V0 if np.any(gross > 0) else 0

                results.append({
                    'lookback_slow': lookback_slow,
                    'gamma': gamma,
                    'conf_threshold': confidence_threshold,
                    'sharpe': round(sharpe, 3),
                    'avg_leverage': round(avg_lev, 1),
                })

                print(f"    lb={lookback_slow}, γ={gamma:.0e}, "
                      f"conf={confidence_threshold}"
                      f" → Sharpe={sharpe:.3f}, Avg Lev={avg_lev:.1f}x")

                # Only consider params where avg leverage ≤ 6x
                # This ensures the strategy doesn't rely on extreme leverage
                if avg_lev <= 6.0 and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print("\n  Top 5 parameter combinations (with avg leverage ≤ 6x):")
    filtered = results_df[results_df['avg_leverage'] <= 6.0]
    print(filtered.head().to_string(index=False))
    print(f"\n  ✅ Best: lookback={best_params['lookback_slow']}, "
          f"γ={best_params['gamma']:.0e}, "
          f"conf_threshold={best_params['confidence_threshold']}, "
          f"Sharpe={best_sharpe:.3f}")

    results_df.to_csv(os.path.join(DATA_DIR, "strategy1_calibration.csv"), index=False)
    return best_params


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: TREND-FOLLOWING (TIME-SERIES MOMENTUM)
# ═══════════════════════════════════════════════════════════════════════

def generate_trend_signals(prices, fast_ma, slow_ma):
    """
    Trend-following signals using dual moving average crossover.

    For each coin at each hour:
    1. Compute fast MA (recent trend, e.g., last 12 hours)
    2. Compute slow MA (longer trend, e.g., last 96 hours)
    3. Signal = (fast - slow) / (slow × volatility)

    Positive signal = uptrend (fast MA above slow MA)
    Negative signal = downtrend (fast MA below slow MA)
    Near zero = no clear trend (MAs are close together)
    """
    close = prices[COINS].values
    n, k = close.shape
    signals = np.full((n, k), np.nan)

    for t in range(slow_ma, n):
        for i in range(k):
            fast = close[t - fast_ma + 1:t + 1, i].mean()
            slow = close[t - slow_ma + 1:t + 1, i].mean()

            # Volatility of this coin over the slow MA window
            rets = np.diff(close[max(0, t - slow_ma):t + 1, i]) / close[max(0, t - slow_ma):t, i]
            vol = rets.std() if len(rets) > 1 else 1e-8

            if vol > 0 and slow > 0:
                signals[t, i] = (fast - slow) / (slow * vol)

    return pd.DataFrame(signals, index=prices.index, columns=COINS)


def backtest_strategy2(prices, returns, params, train_end, rebal_freq=720):
    """
    Backtest trend-following with TWO-STEP position sizing.

    rebal_freq: hours between rebalances
        - 6 for calibration (more data points for Sharpe estimation)
        - 720 (monthly) for production (lower costs)
        Reference: Moskowitz, Ooi & Pedersen (2012) use monthly rebalancing

    Reference: Moskowitz, Ooi & Pedersen (2012)
    """
    fast_ma = params['fast_ma']
    slow_ma = params['slow_ma']
    vol_lookback = params['vol_lookback']
    position_scale = params['position_scale']
    confidence_threshold = params.get('confidence_threshold', 2.0)

    n = len(prices)
    close = prices[COINS].values
    returns_vals = returns[COINS].values
    signals = generate_trend_signals(prices, fast_ma, slow_ma)
    signals_vals = signals.values

    theta_arr = np.zeros((n, len(COINS)))
    start_idx = slow_ma + 1
    REBAL_FREQ = rebal_freq

    for t in range(start_idx, n):
        # NOT rebalancing → let positions drift with the market
        # Just HOLDING, not trading → no transaction costs
        if t % REBAL_FREQ != 0:
            if t > 0:
                theta_arr[t] = theta_arr[t - 1] * (1 + returns_vals[t])
            else:
                theta_arr[t] = np.zeros(len(COINS))
            # Hard cap: if drift pushes gross exposure above $100k, scale down
            gross_drift = np.sum(np.abs(theta_arr[t]))
            if gross_drift > MAX_GROSS_EXPOSURE:
                theta_arr[t] *= MAX_GROSS_EXPOSURE / gross_drift
            continue

        s_t = signals_vals[t]

        if np.all(np.isnan(s_t)):
            if t > 0:
                theta_arr[t] = theta_arr[t - 1]
            continue

        s_t = np.nan_to_num(s_t, 0.0)

        # ── STEP 1: Direction & relative weights ──
        # Inverse volatility weighting (risk parity)
        # Riskier coins get SMALLER positions
        inv_vols = np.ones(len(COINS))
        for i in range(len(COINS)):
            rets = np.diff(close[max(0, t - vol_lookback):t + 1, i]) / close[max(0, t - vol_lookback):t, i]
            vol = rets.std() if len(rets) > 1 else 1e-8
            inv_vols[i] = 1.0 / vol if vol > 0 else 0.0

        # Normalise inverse vols so they sum to len(COINS)
        inv_vol_sum = inv_vols.sum()
        if inv_vol_sum > 0:
            inv_vols = inv_vols / inv_vol_sum * len(COINS)

        # Raw weights: signal × inv_vol (captures direction + risk parity)
        raw_weights = s_t * inv_vols * position_scale

        # Normalise to unit gross exposure (so we can scale separately)
        gross_raw = np.sum(np.abs(raw_weights))
        if gross_raw > 0:
            unit_weights = raw_weights / gross_raw
        else:
            theta_arr[t] = 0.0
            continue

        # ── STEP 2: Signal confidence → total exposure ──
        #
        # signal_confidence = how strong are the trend signals right now?
        # avg|signal| = 0 → no trends → stay at base capital ($10k)
        # avg|signal| = 1 → moderate trends → some leverage
        # avg|signal| ≥ threshold → strong trends → up to max leverage
        #
        # target_exposure = V0 + (MAX - V0) × min(1, confidence / threshold)
        #
        # Example with threshold = 2.0:
        #   confidence = 0.0 → target = $10k (1x)
        #   confidence = 0.5 → target = $32.5k (3.25x)
        #   confidence = 1.0 → target = $55k (5.5x)
        #   confidence = 2.0 → target = $100k (10x)

        signal_confidence = np.mean(np.abs(s_t))
        leverage_fraction = min(1.0, signal_confidence / confidence_threshold)
        target_exposure = V0 + (MAX_GROSS_EXPOSURE - V0) * leverage_fraction

        # Scale unit weights to target exposure
        theta_t = unit_weights * target_exposure

        # Hard cap: never exceed $100k (safety net)
        gross_final = np.sum(np.abs(theta_t))
        if gross_final > MAX_GROSS_EXPOSURE:
            theta_t *= MAX_GROSS_EXPOSURE / gross_final

        theta_arr[t] = theta_t

    theta = pd.DataFrame(theta_arr, index=prices.index, columns=COINS)
    return theta, signals


def calibrate_strategy2(prices_train, returns_train):
    """
    Grid search for Strategy 2 (trend-following) parameters.

    Parameters:
    - fast_ma: fast moving average window (hours)
    - slow_ma: slow moving average window (hours)
    - confidence_threshold: controls how signal strength maps to leverage.
      Crypto trend signals are typically large (avg |signal| = 3-8),
      so we need high thresholds to avoid always being at max leverage:
        threshold=3  → still aggressive, often near 10x
        threshold=6  → moderate, average leverage ~5x
        threshold=10 → conservative, average leverage ~3x
        threshold=15 → very conservative, mostly near 1x

    Note: position_scale is fixed at 1.0 because the two-step approach
    normalises weights before scaling — the scale factor cancels out.
    Only the confidence_threshold controls effective leverage.

    Selection: pick highest in-sample Sharpe among all parameter combos.
    """
    print("\n  Calibrating Strategy 2 parameters (grid search)...")

    best_sharpe = -np.inf
    best_params = None
    results = []

    returns_vals = returns_train[COINS].values

    for fast_ma in [12, 24, 48]:
        for slow_ma in [96, 168, 336]:
            if fast_ma >= slow_ma:
                continue
            for confidence_threshold in [3.0, 5.0, 8.0, 12.0, 18.0]:
                params = {
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'vol_lookback': 168,
                    'position_scale': 1.0,  # fixed (normalised away)
                    'confidence_threshold': confidence_threshold,
                }

                theta, _ = backtest_strategy2(
                    prices_train, returns_train, params, TRAIN_END,
                    rebal_freq=6  # Use 6h for calibration (more data points)
                )

                theta_vals = theta[COINS].values
                start = slow_ma + 2
                pnl = np.nansum(theta_vals[start:-1] * returns_vals[start + 1:], axis=1)

                if len(pnl) == 0 or np.std(pnl) == 0:
                    continue

                sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(24 * 365)

                # Track average leverage
                gross = np.sum(np.abs(theta_vals[start:]), axis=1)
                avg_lev = np.mean(gross[gross > 0]) / V0 if np.any(gross > 0) else 0

                results.append({
                    'fast_ma': fast_ma, 'slow_ma': slow_ma,
                    'conf_threshold': confidence_threshold,
                    'sharpe': round(sharpe, 3),
                    'avg_leverage': round(avg_lev, 1),
                })

                print(f"    fast={fast_ma}, slow={slow_ma}, "
                      f"conf={confidence_threshold}"
                      f" → Sharpe={sharpe:.3f}, Avg Lev={avg_lev:.1f}x")

                # Only consider params where avg leverage ≤ 6x
                # This ensures the strategy doesn't rely on extreme leverage
                if avg_lev <= 6.0 and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print("\n  Top 5 parameter combinations (with avg leverage ≤ 6x):")
    filtered = results_df[results_df['avg_leverage'] <= 6.0]
    print(filtered.head().to_string(index=False))
    print(f"\n  ✅ Best: fast={best_params['fast_ma']}, slow={best_params['slow_ma']}, "
          f"conf_threshold={best_params['confidence_threshold']}, "
          f"Sharpe={best_sharpe:.3f}")

    results_df.to_csv(os.path.join(DATA_DIR, "strategy2_calibration.csv"), index=False)
    return best_params


# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE PREVIEW
# ═══════════════════════════════════════════════════════════════════════

def quick_performance(theta, returns, name, train_end):
    """Quick performance stats with leverage info."""
    pnl = (theta.shift(1) * returns).sum(axis=1).dropna()
    gross = theta.abs().sum(axis=1)

    pnl_is = pnl.loc[:train_end]
    pnl_oos = pnl.loc[train_end:]
    gross_all = gross[gross > 0]

    def show(p, label):
        if len(p) == 0 or p.std() == 0:
            print(f"    {label}: No trades")
            return
        sharpe = p.mean() / p.std() * np.sqrt(24 * 365)
        cum = p.sum()
        dd = (p.cumsum() - p.cumsum().cummax()).min()
        print(f"    {label}: Sharpe={sharpe:.3f}, PnL=${cum:,.0f}, MaxDD=${dd:,.0f}")

    print(f"\n  {name}:")
    show(pnl_is, "In-sample ")
    show(pnl_oos, "Out-of-sample")
    print(f"    Leverage: avg={gross_all.mean()/V0:.1f}x, "
          f"max={gross_all.max()/V0:.1f}x, "
          f"min={gross_all[gross_all>0].min()/V0:.1f}x")
    return pnl


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_strategy1(theta, signals, prices, returns, train_end):
    """Diagnostic plots for Strategy 1: Mean-Reversion MVO."""
    colors = {'BTCUSDT': '#F7931A', 'ETHUSDT': '#627EEA', 'DOGEUSDT': '#C2A633',
              'BNBUSDT': '#F3BA2F', 'XRPUSDT': '#00AAE4'}

    fig = plt.figure(figsize=(14, 17))
    fig.suptitle('Strategy 1: Multi-Asset Mean-Reversion with MVO',
                 fontsize=14, fontweight='bold')
    # First 4 panels share x-axis (datetime); 5th is a histogram (separate)
    gs = fig.add_gridspec(5, 1, hspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4])  # histogram — independent x-axis
    axes = [ax0, ax1, ax2, ax3, ax4]

    ax = axes[0]
    for c in COINS:
        ax.plot(signals.index, signals[c], linewidth=0.3, alpha=0.7,
                color=colors[c], label=c.replace('USDT', ''))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Signal'); ax.set_title('Mean-Reversion Signals (Avellaneda-Lee)')
    ax.legend(fontsize=7, ncol=5); ax.grid(True, alpha=0.3); ax.set_ylim(-4, 4)

    ax = axes[1]
    for c in COINS:
        ax.plot(theta.index, theta[c] / 1000, linewidth=0.5, alpha=0.8,
                color=colors[c], label=c.replace('USDT', ''))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Position ($1000)'); ax.set_title('MVO-Optimised Positions')
    ax.legend(fontsize=7, ncol=5); ax.grid(True, alpha=0.3)

    ax = axes[2]
    gross = theta.abs().sum(axis=1)
    leverage = gross / V0
    ax.plot(theta.index, leverage, linewidth=0.5, color='purple')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10x Max')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1x (no leverage)')
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Leverage (x)'); ax.set_title('Dynamic Leverage Over Time')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3]
    pnl = (theta.shift(1) * returns).sum(axis=1).dropna().cumsum()
    ax.plot(pnl.index, pnl, linewidth=1, color='darkgreen')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7, label='Train/Test')
    ax.fill_between(pnl.index, 0, pnl, where=pnl >= 0, alpha=0.15, color='green')
    ax.fill_between(pnl.index, 0, pnl, where=pnl < 0, alpha=0.15, color='red')
    ax.set_ylabel('Cum. PnL (USDT)'); ax.set_title('Gross Cumulative PnL')
    ax.set_xlabel('Date')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Leverage histogram (independent x-axis)
    ax = axes[4]
    lev_nonzero = leverage[leverage > 0]
    ax.hist(lev_nonzero, bins=50, color='purple', alpha=0.7, edgecolor='white')
    ax.axvline(x=lev_nonzero.mean(), color='red', linestyle='--',
               label=f'Avg: {lev_nonzero.mean():.1f}x')
    ax.set_xlabel('Leverage (x)'); ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Leverage Used')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig5_strategy1_mr.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig5_strategy1_mr.png")
    plt.close()


def plot_strategy2(theta, signals, prices, returns, train_end, best_params):
    """Diagnostic plots for Strategy 2: Trend-Following."""
    colors = {'BTCUSDT': '#F7931A', 'ETHUSDT': '#627EEA', 'DOGEUSDT': '#C2A633',
              'BNBUSDT': '#F3BA2F', 'XRPUSDT': '#00AAE4'}

    fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
    fig.suptitle('Strategy 2: Trend-Following (Time-Series Momentum)',
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    for c in COINS:
        norm = prices[c] / prices[c].iloc[0] * 100
        ax.plot(prices.index, norm, linewidth=0.7, color=colors[c],
                label=c.replace('USDT', ''))
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Normalised (100)'); ax.set_title('Normalised Prices')
    ax.legend(fontsize=7, ncol=5); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in COINS:
        ax.plot(signals.index, signals[c], linewidth=0.3, alpha=0.7,
                color=colors[c], label=c.replace('USDT', ''))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Signal'); ax.set_title(f'Trend Signals (Fast={best_params["fast_ma"]}h, Slow={best_params["slow_ma"]}h)')
    ax.legend(fontsize=7, ncol=5); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for c in COINS:
        ax.plot(theta.index, theta[c] / 1000, linewidth=0.5, alpha=0.8,
                color=colors[c], label=c.replace('USDT', ''))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Position ($1000)'); ax.set_title('Volatility-Scaled Positions')
    ax.legend(fontsize=7, ncol=5); ax.grid(True, alpha=0.3)

    ax = axes[3]
    gross = theta.abs().sum(axis=1)
    leverage = gross / V0
    ax.plot(theta.index, leverage, linewidth=0.5, color='purple')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10x Max')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1x (no leverage)')
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Leverage (x)'); ax.set_title('Dynamic Leverage Over Time')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[4]
    pnl = (theta.shift(1) * returns).sum(axis=1).dropna().cumsum()
    ax.plot(pnl.index, pnl, linewidth=1, color='darkgreen')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7, label='Train/Test')
    ax.fill_between(pnl.index, 0, pnl, where=pnl >= 0, alpha=0.15, color='green')
    ax.fill_between(pnl.index, 0, pnl, where=pnl < 0, alpha=0.15, color='red')
    ax.set_ylabel('Cum. PnL (USDT)'); ax.set_xlabel('Date')
    ax.set_title('Gross Cumulative PnL (before transaction costs)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig6_strategy2_tf.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig6_strategy2_tf.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  COMP0051 - TASK 2: TRADING STRATEGIES")
    print("═" * 60)

    # Load data
    print("\n  Loading data...")
    prices, returns = load_all_data()
    print(f"  Loaded: {len(prices)} bars, {len(COINS)} coins")
    print(f"  Range: {prices.index[0]} → {prices.index[-1]}")

    prices_train = prices.loc[:TRAIN_END]
    returns_train = returns.loc[:TRAIN_END]
    print(f"  Train: {len(prices_train)} | Test: {len(prices) - len(prices_train)}")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 1: MEAN-REVERSION + MVO
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STRATEGY 1: Multi-Asset Mean-Reversion with MVO")
    print("=" * 60)

    best_s1 = calibrate_strategy1(prices_train, returns_train)

    print("\n  Running full backtest (train + test)...")
    theta_s1, sig_s1 = backtest_strategy1(prices, returns, best_s1, TRAIN_END)
    theta_s1.to_parquet(os.path.join(DATA_DIR, "strategy1_positions.parquet"))
    print("  ✅ Saved strategy1_positions.parquet")
    quick_performance(theta_s1, returns, "Strategy 1 (Mean-Reversion + MVO)", TRAIN_END)
    plot_strategy1(theta_s1, sig_s1, prices, returns, TRAIN_END)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 2: TREND-FOLLOWING
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STRATEGY 2: Trend-Following (Time-Series Momentum)")
    print("=" * 60)

    best_s2 = calibrate_strategy2(prices_train, returns_train)

    print("\n  Running full backtest (train + test)...")
    theta_s2, sig_s2 = backtest_strategy2(prices, returns, best_s2, TRAIN_END)
    theta_s2.to_parquet(os.path.join(DATA_DIR, "strategy2_positions.parquet"))
    print("  ✅ Saved strategy2_positions.parquet")
    quick_performance(theta_s2, returns, "Strategy 2 (Trend-Following)", TRAIN_END)
    plot_strategy2(theta_s2, sig_s2, prices, returns, TRAIN_END, best_s2)

    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TASK 2 COMPLETE!")
    print("  Strategy 1: Mean-Reversion + MVO (Avellaneda-Lee / Markowitz)")
    print("  Strategy 2: Trend-Following (Moskowitz-Ooi-Pedersen)")
    print("  → Proceed to Task 3 (Transaction Costs)")
    print("=" * 60)
