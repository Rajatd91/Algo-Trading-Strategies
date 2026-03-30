"""
COMP0051 - Rebalancing Frequency Comparison

Tests different rebalancing frequencies for both strategies:
  6h, 12h, 24h (daily), 48h, 168h (weekly)

For each frequency, we:
  1. Re-run the backtest with that rebalancing period
  2. Compute transaction costs using the calibrated Roll model
  3. Compute OOS Sharpe (net of costs) and total net PnL

The goal: find the frequency that maximises NET performance.

Why this matters:
  - Rebalancing too often → high turnover → costs eat profits
  - Rebalancing too rarely → stale signals → miss opportunities
  - There's an optimal middle ground

Reference: Cartea, Jaimungal & Penalva (2015) Ch. 7 discusses
optimal execution frequency and the turnover-alpha tradeoff.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project directory to path so we can import from strategy file
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

COINS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT']
V0 = 10_000
MAX_GROSS_EXPOSURE = 100_000
TRAIN_END = pd.Timestamp('2024-12-31 23:00:00')

# Calibrated Roll model slippage (from Task 3)
SLIPPAGE = {
    'BTCUSDT': 0.000500,   # 5 bps
    'ETHUSDT': 0.000500,   # 5 bps
    'DOGEUSDT': 0.001614,  # 16 bps
    'BNBUSDT': 0.000691,   # 7 bps
    'XRPUSDT': 0.001770,   # 18 bps
}

# Rebalancing frequencies to test (in hours)
# Based on academic literature:
#   - 24h (daily): Avellaneda & Lee (2010) use daily rebalancing
#   - 168h (weekly): common in portfolio management practice
#   - 336h (bi-weekly / 14 days): between weekly and monthly
#   - 720h (monthly / ~30 days): Moskowitz, Ooi & Pedersen (2012) use monthly
#   - 6h: our current setting (for comparison)
REBAL_FREQS = [6, 24, 168, 336, 720]
FREQ_LABELS = ['6h (current)', '24h (daily)', '168h (weekly)', '336h (bi-weekly)', '720h (monthly)']


def load_data():
    prices = pd.DataFrame()
    returns = pd.DataFrame()
    for coin in COINS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet"))
        dr = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_returns.parquet"))
        prices[coin] = df['close']
        returns[coin] = dr['excess_return']
    idx = prices.index.intersection(returns.index)
    return prices.loc[idx], returns.loc[idx]


# ═══════════════════════════════════════════════════════════════════════
# Import signal generators from 2_strategy.py
# ═══════════════════════════════════════════════════════════════════════

def generate_mr_signals(prices, lookback_slow):
    """Mean-reversion signals (Avellaneda-Lee)."""
    close = prices[COINS].values
    n, k = close.shape
    signals = np.full((n, k), np.nan)
    for t in range(lookback_slow, n):
        for i in range(k):
            window = close[t - lookback_slow:t + 1, i]
            ma = window[:-1].mean()
            rets = np.diff(window[:-1]) / window[:-2]
            vol = rets.std() if len(rets) > 1 else 1e-8
            if vol > 0 and close[t, i] > 0:
                signals[t, i] = (ma - close[t, i]) / (close[t, i] * vol)
    for t in range(n):
        row = signals[t]
        if np.all(np.isnan(row)):
            continue
        mu = np.nanmean(row)
        sigma = np.nanstd(row)
        if sigma > 0:
            signals[t] = (row - mu) / sigma
    return pd.DataFrame(signals, index=prices.index, columns=COINS)


def generate_trend_signals(prices, fast_ma, slow_ma):
    """Trend-following signals (dual MA crossover)."""
    close = prices[COINS].values
    n, k = close.shape
    signals = np.full((n, k), np.nan)
    for t in range(slow_ma, n):
        for i in range(k):
            fast = close[t - fast_ma + 1:t + 1, i].mean()
            slow = close[t - slow_ma + 1:t + 1, i].mean()
            rets = np.diff(close[max(0, t - slow_ma):t + 1, i]) / close[max(0, t - slow_ma):t, i]
            vol = rets.std() if len(rets) > 1 else 1e-8
            if vol > 0 and slow > 0:
                signals[t, i] = (fast - slow) / (slow * vol)
    return pd.DataFrame(signals, index=prices.index, columns=COINS)


# ═══════════════════════════════════════════════════════════════════════
# MVO optimizer (simplified — same as in 2_strategy.py)
# ═══════════════════════════════════════════════════════════════════════

from scipy.optimize import minimize

def mvo_optimize(mu, sigma, theta_prev, gamma, tc_penalty, max_gross):
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


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST FUNCTIONS WITH CONFIGURABLE REBAL FREQ
# ═══════════════════════════════════════════════════════════════════════

def backtest_s1(prices, returns, rebal_freq):
    """Strategy 1: MR+MVO with given rebalancing frequency."""
    # Best params from calibration
    lookback_slow = 168
    lookback_cov = 168
    gamma = 5e-4
    kappa = 0.01
    tc_penalty = 0.001
    confidence_threshold = 2.0

    signals = generate_mr_signals(prices, lookback_slow)
    signals_vals = signals.values
    returns_vals = returns[COINS].values
    n = len(prices)

    theta_arr = np.zeros((n, len(COINS)))
    theta_prev = np.zeros(len(COINS))
    start_idx = max(lookback_slow, lookback_cov) + 1
    last_sigma = np.eye(len(COINS)) * 1e-4

    for t in range(start_idx, n):
        if t % rebal_freq != 0:
            if t > 0:
                theta_arr[t] = theta_arr[t - 1] * (1 + returns_vals[t])
            else:
                theta_arr[t] = theta_prev
            # Hard cap: if drift pushes gross exposure above $100k, scale down
            gross_drift = np.sum(np.abs(theta_arr[t]))
            if gross_drift > MAX_GROSS_EXPOSURE:
                theta_arr[t] *= MAX_GROSS_EXPOSURE / gross_drift
            continue

        s_t = signals_vals[t]
        if np.all(np.isnan(s_t)) or np.all(s_t == 0):
            theta_arr[t] = theta_prev * 0.9
            theta_prev = theta_arr[t].copy()
            continue

        s_t = np.nan_to_num(s_t, 0.0)
        mu_t = kappa * s_t

        ret_window = returns_vals[max(0, t - lookback_cov):t]
        if len(ret_window) >= 20:
            cov_raw = np.cov(ret_window, rowvar=False)
            if not (np.any(np.isnan(cov_raw)) or np.any(np.isinf(cov_raw))):
                diag = np.diag(np.diag(cov_raw))
                last_sigma = 0.5 * cov_raw + 0.5 * diag

        theta_raw = mvo_optimize(mu_t, last_sigma, theta_prev, gamma,
                                  tc_penalty, MAX_GROSS_EXPOSURE)

        signal_confidence = np.mean(np.abs(s_t))
        leverage_fraction = min(1.0, signal_confidence / confidence_threshold)
        target_exposure = V0 + (MAX_GROSS_EXPOSURE - V0) * leverage_fraction

        gross_raw = np.sum(np.abs(theta_raw))
        if gross_raw > 0:
            theta_scaled = theta_raw * (target_exposure / gross_raw)
        else:
            theta_scaled = theta_raw

        gross_final = np.sum(np.abs(theta_scaled))
        if gross_final > MAX_GROSS_EXPOSURE:
            theta_scaled *= MAX_GROSS_EXPOSURE / gross_final

        theta_arr[t] = theta_scaled
        theta_prev = theta_scaled.copy()

    return pd.DataFrame(theta_arr, index=prices.index, columns=COINS)


def backtest_s2(prices, returns, rebal_freq):
    """Strategy 2: Trend-following with given rebalancing frequency."""
    # Best params from calibration
    fast_ma = 12
    slow_ma = 96
    vol_lookback = 168
    position_scale = 1.0
    confidence_threshold = 18.0

    signals = generate_trend_signals(prices, fast_ma, slow_ma)
    signals_vals = signals.values
    returns_vals = returns[COINS].values
    close = prices[COINS].values
    n = len(prices)

    theta_arr = np.zeros((n, len(COINS)))
    start_idx = slow_ma + 1

    for t in range(start_idx, n):
        if t % rebal_freq != 0:
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

        inv_vols = np.ones(len(COINS))
        for i in range(len(COINS)):
            rets = np.diff(close[max(0, t - vol_lookback):t + 1, i]) / close[max(0, t - vol_lookback):t, i]
            vol = rets.std() if len(rets) > 1 else 1e-8
            inv_vols[i] = 1.0 / vol if vol > 0 else 0.0
        inv_vol_sum = inv_vols.sum()
        if inv_vol_sum > 0:
            inv_vols = inv_vols / inv_vol_sum * len(COINS)

        raw_weights = s_t * inv_vols * position_scale
        gross_raw = np.sum(np.abs(raw_weights))
        if gross_raw > 0:
            unit_weights = raw_weights / gross_raw
        else:
            theta_arr[t] = 0.0
            continue

        signal_confidence = np.mean(np.abs(s_t))
        leverage_fraction = min(1.0, signal_confidence / confidence_threshold)
        target_exposure = V0 + (MAX_GROSS_EXPOSURE - V0) * leverage_fraction

        theta_t = unit_weights * target_exposure
        gross_final = np.sum(np.abs(theta_t))
        if gross_final > MAX_GROSS_EXPOSURE:
            theta_t *= MAX_GROSS_EXPOSURE / gross_final

        theta_arr[t] = theta_t

    return pd.DataFrame(theta_arr, index=prices.index, columns=COINS)


# ═══════════════════════════════════════════════════════════════════════
# COST & PERFORMANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_costs(theta, returns):
    """Transaction costs using calibrated Roll slippage."""
    n = len(theta)
    costs = np.zeros(n)
    for coin in COINS:
        s = SLIPPAGE[coin]
        pos = theta[coin].values
        ret = returns[coin].values
        for t in range(1, n):
            old_grown = pos[t - 1] * (1 + ret[t])
            costs[t] += s * abs(pos[t] - old_grown)
    return pd.Series(costs, index=theta.index)


def evaluate(theta, returns, costs, train_end):
    """Compute key metrics for IS and OOS."""
    pnl_gross = (theta.shift(1) * returns).sum(axis=1).fillna(0)
    pnl_net = pnl_gross - costs

    results = {}
    for label, mask in [('IS', pnl_net.index <= train_end),
                        ('OOS', pnl_net.index > train_end),
                        ('Full', slice(None))]:
        p_g = pnl_gross.loc[mask]
        p_n = pnl_net.loc[mask]
        c = costs.loc[mask]

        if len(p_n) == 0 or p_n.std() == 0:
            results[label] = {'sharpe_gross': 0, 'sharpe_net': 0,
                              'pnl_gross': 0, 'pnl_net': 0, 'costs': 0,
                              'turnover': 0, 'holding_hrs': 0}
            continue

        sharpe_g = p_g.mean() / p_g.std() * np.sqrt(24*365) if p_g.std() > 0 else 0
        sharpe_n = p_n.mean() / p_n.std() * np.sqrt(24*365)

        # Turnover
        theta_p = theta.loc[p_n.index]
        returns_p = returns.loc[p_n.index]
        total_traded = 0
        for coin in COINS:
            pos = theta_p[coin].values
            ret = returns_p[coin].values
            for t in range(1, len(pos)):
                total_traded += abs(pos[t] - pos[t-1] * (1 + ret[t]))
        hrs = len(p_n)
        turnover = total_traded / (V0 * hrs) if hrs > 0 else 0
        avg_pos = theta_p.abs().sum(axis=1).mean()
        avg_hourly_to = total_traded / hrs if hrs > 0 else 1
        holding = avg_pos / avg_hourly_to if avg_hourly_to > 0 else 0

        results[label] = {
            'sharpe_gross': round(sharpe_g, 3),
            'sharpe_net': round(sharpe_n, 3),
            'pnl_gross': round(p_g.sum(), 0),
            'pnl_net': round(p_n.sum(), 0),
            'costs': round(c.sum(), 0),
            'turnover': round(turnover, 4),
            'holding_hrs': round(holding, 1),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN: RUN COMPARISON
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  REBALANCING FREQUENCY COMPARISON")
    print("  Testing: 6h, 12h, 24h, 48h, 168h (weekly)")
    print("═" * 65)

    prices, returns = load_data()
    print(f"  Data: {len(prices)} hours, {len(COINS)} coins")

    # ── Strategy 1: Mean-Reversion + MVO ──
    print("\n" + "=" * 65)
    print("  STRATEGY 1: Mean-Reversion + MVO")
    print("=" * 65)

    s1_results = []
    for freq, label in zip(REBAL_FREQS, FREQ_LABELS):
        print(f"\n  Testing rebal every {label}...", end=" ", flush=True)
        theta = backtest_s1(prices, returns, freq)
        costs = compute_costs(theta, returns)
        metrics = evaluate(theta, returns, costs, TRAIN_END)
        print(f"OOS Net Sharpe = {metrics['OOS']['sharpe_net']:.3f}, "
              f"Net PnL = ${metrics['OOS']['pnl_net']:,.0f}")

        s1_results.append({
            'rebal_freq': freq,
            'label': label,
            'oos_sharpe_gross': metrics['OOS']['sharpe_gross'],
            'oos_sharpe_net': metrics['OOS']['sharpe_net'],
            'oos_pnl_gross': metrics['OOS']['pnl_gross'],
            'oos_pnl_net': metrics['OOS']['pnl_net'],
            'oos_costs': metrics['OOS']['costs'],
            'turnover': metrics['OOS']['turnover'],
            'holding_hrs': metrics['OOS']['holding_hrs'],
            'full_pnl_net': metrics['Full']['pnl_net'],
            'full_sharpe_net': metrics['Full']['sharpe_net'],
        })

    s1_df = pd.DataFrame(s1_results)
    print("\n  ┌─────────────┬────────────┬───────────┬────────────┬──────────┬──────────────┐")
    print("  │ Rebal Freq  │ OOS Sharpe │ OOS Sharpe│ OOS Net PnL│ OOS Costs│ Holding (hrs)│")
    print("  │             │  (gross)   │  (net)    │            │          │              │")
    print("  ├─────────────┼────────────┼───────────┼────────────┼──────────┼──────────────┤")
    for _, r in s1_df.iterrows():
        best_flag = " ◄" if r['oos_sharpe_net'] == s1_df['oos_sharpe_net'].max() else ""
        print(f"  │ {r['label']:>11s} │   {r['oos_sharpe_gross']:>7.3f}  │  {r['oos_sharpe_net']:>7.3f}  │"
              f" ${r['oos_pnl_net']:>9,.0f}│ ${r['oos_costs']:>7,.0f} │  {r['holding_hrs']:>10.1f}  │{best_flag}")
    print("  └─────────────┴────────────┴───────────┴────────────┴──────────┴──────────────┘")

    best_s1 = s1_df.loc[s1_df['oos_sharpe_net'].idxmax()]
    print(f"\n  ✅ Best for S1: {best_s1['label']} → OOS Net Sharpe = {best_s1['oos_sharpe_net']:.3f}")

    # ── Strategy 2: Trend-Following ──
    print("\n" + "=" * 65)
    print("  STRATEGY 2: Trend-Following")
    print("=" * 65)

    s2_results = []
    for freq, label in zip(REBAL_FREQS, FREQ_LABELS):
        print(f"\n  Testing rebal every {label}...", end=" ", flush=True)
        theta = backtest_s2(prices, returns, freq)
        costs = compute_costs(theta, returns)
        metrics = evaluate(theta, returns, costs, TRAIN_END)
        print(f"OOS Net Sharpe = {metrics['OOS']['sharpe_net']:.3f}, "
              f"Net PnL = ${metrics['OOS']['pnl_net']:,.0f}")

        s2_results.append({
            'rebal_freq': freq,
            'label': label,
            'oos_sharpe_gross': metrics['OOS']['sharpe_gross'],
            'oos_sharpe_net': metrics['OOS']['sharpe_net'],
            'oos_pnl_gross': metrics['OOS']['pnl_gross'],
            'oos_pnl_net': metrics['OOS']['pnl_net'],
            'oos_costs': metrics['OOS']['costs'],
            'turnover': metrics['OOS']['turnover'],
            'holding_hrs': metrics['OOS']['holding_hrs'],
            'full_pnl_net': metrics['Full']['pnl_net'],
            'full_sharpe_net': metrics['Full']['sharpe_net'],
        })

    s2_df = pd.DataFrame(s2_results)
    print("\n  ┌─────────────┬────────────┬───────────┬────────────┬──────────┬──────────────┐")
    print("  │ Rebal Freq  │ OOS Sharpe │ OOS Sharpe│ OOS Net PnL│ OOS Costs│ Holding (hrs)│")
    print("  │             │  (gross)   │  (net)    │            │          │              │")
    print("  ├─────────────┼────────────┼───────────┼────────────┼──────────┼──────────────┤")
    for _, r in s2_df.iterrows():
        best_flag = " ◄" if r['oos_sharpe_net'] == s2_df['oos_sharpe_net'].max() else ""
        print(f"  │ {r['label']:>11s} │   {r['oos_sharpe_gross']:>7.3f}  │  {r['oos_sharpe_net']:>7.3f}  │"
              f" ${r['oos_pnl_net']:>9,.0f}│ ${r['oos_costs']:>7,.0f} │  {r['holding_hrs']:>10.1f}  │{best_flag}")
    print("  └─────────────┴────────────┴───────────┴────────────┴──────────┴──────────────┘")

    best_s2 = s2_df.loc[s2_df['oos_sharpe_net'].idxmax()]
    print(f"\n  ✅ Best for S2: {best_s2['label']} → OOS Net Sharpe = {best_s2['oos_sharpe_net']:.3f}")

    # ── Plot comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rebalancing Frequency Comparison (Out-of-Sample, Net of Costs)',
                 fontsize=13, fontweight='bold')

    for row, (df, sname) in enumerate([(s1_df, 'Strategy 1: MR+MVO'),
                                        (s2_df, 'Strategy 2: Trend-Following')]):
        # Net Sharpe
        ax = axes[row, 0]
        colors = ['green' if s > 0 else 'red' for s in df['oos_sharpe_net']]
        bars = ax.bar(range(len(df)), df['oos_sharpe_net'], color=colors, alpha=0.8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['label'], rotation=45)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('OOS Net Sharpe')
        ax.set_title(f'{sname}\nNet Sharpe Ratio')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(df['oos_sharpe_net']):
            ax.text(i, v + 0.02 if v >= 0 else v - 0.04,
                    f'{v:.2f}', ha='center', fontsize=9)

        # Net PnL and Costs
        ax = axes[row, 1]
        x = range(len(df))
        ax.bar(x, df['oos_pnl_gross'], width=0.35, label='Gross PnL',
               color='steelblue', alpha=0.7, align='center')
        ax.bar([xi + 0.35 for xi in x], -df['oos_costs'], width=0.35,
               label='Costs', color='red', alpha=0.7, align='center')
        ax.set_xticks([xi + 0.175 for xi in x])
        ax.set_xticklabels(df['label'], rotation=45)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('USDT')
        ax.set_title(f'{sname}\nGross PnL vs Costs')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig9_rebal_comparison.png'), dpi=150, bbox_inches='tight')
    print("\n  ✅ Saved fig9_rebal_comparison.png")
    plt.close()

    # Save results
    s1_df.to_csv(os.path.join(DATA_DIR, 'rebal_comparison_s1.csv'), index=False)
    s2_df.to_csv(os.path.join(DATA_DIR, 'rebal_comparison_s2.csv'), index=False)
    print("  ✅ Saved rebal_comparison CSVs")

    print("\n" + "=" * 65)
    print(f"  RECOMMENDATION:")
    print(f"  Strategy 1: rebalance every {best_s1['label']}")
    print(f"  Strategy 2: rebalance every {best_s2['label']}")
    print(f"  (Based on highest OOS Net Sharpe)")
    print("=" * 65)
