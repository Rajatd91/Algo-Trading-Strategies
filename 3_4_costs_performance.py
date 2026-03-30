"""
COMP0051 - Algorithmic Trading Coursework
Task 3: Transaction Costs
Task 4: Performance Metrics

TWO slippage estimation methods:

1. Roll (1984) model:  s = sqrt( -Cov(Δp_t, Δp_{t-1}) )
   - Classic method, but designed for tick-by-tick data
   - On hourly bars, it overestimates because it picks up ALL mean-reversion
     patterns, not just the bid-ask bounce
   - Reference: Roll (1984) "A Simple Implicit Measure of the Effective
     Bid-Ask Spread in an Efficient Market", Journal of Finance

2. Corwin-Schultz (2012) high-low estimator:
   - Uses High and Low prices from bars (not just close)
   - Designed for bar data (daily/hourly) — more appropriate for our case
   - Based on the idea that high-low range reflects both volatility AND spread
   - By comparing single-bar range to two-bar range, it isolates the spread
   - Reference: Corwin & Schultz (2012) "A Simple Way to Estimate Bid-Ask
     Spreads from Daily High and Low Prices", Journal of Finance

We compute BOTH and compare. The Corwin-Schultz estimate is used for
the main performance computation since it's more appropriate for hourly bars.

Task 4 formula from the coursework:
    ΔV_t = Σ(θ_t^i × r_t^i) - Cost_t
    Cost_t = s × Σ|θ_t^i - θ_{t-1}^i × (1 + r_{t-1}^i)|
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
COINS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT']
COIN_LABELS = ['BTC', 'ETH', 'DOGE', 'BNB', 'XRP']

V0 = 10_000
TRAIN_END = pd.Timestamp('2024-12-31 23:00:00')


# ═══════════════════════════════════════════════════════════════════════
# TASK 3: SLIPPAGE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

def estimate_roll_slippage():
    """
    Roll (1984) model on hourly price changes, calibrated to Binance costs.

    TWO-STEP approach:

    Step 1: Compute raw Roll estimates on hourly Δp (price changes)
        s_raw = sqrt( -Cov(Δp_t, Δp_{t-1}) ) / avg_price
    This gives RELATIVE spreads across coins (BTC tightest, DOGE widest).

    Step 2: Calibrate to Binance's known fee structure
        On hourly data, the Roll model overestimates absolute spread because
        it captures mean-reversion beyond the bid-ask bounce. We calibrate
        the absolute level to Binance's actual cost (~10 bps average):
            s_calibrated_i = (s_raw_i / mean(s_raw)) × 10 bps
        This preserves the Roll model's cross-asset ordering while matching
        realistic trading costs.

    Why 10 bps average?
        Binance spot: 10 bps maker/taker fee + ~0-5 bps spread
        = ~10-15 bps total per trade
        We use 10 bps as a conservative estimate (before spread).

    Reference: Roll (1984), Journal of Finance, 39(4), pp. 1127-1139.
    Also: Bouchaud et al. (2018), Tsay (2010) Ch. 5.
    """
    BINANCE_AVG_COST = 0.0010  # 10 bps = Binance maker/taker fee
    results = {}

    # Step 1: Raw Roll estimates on hourly price changes
    raw_estimates = {}
    for coin in COINS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet"))
        close = df['close'].values

        dp = np.diff(close)
        train_mask = df.index <= TRAIN_END
        n_train = train_mask.sum()

        dp_train = dp[:n_train - 1]
        dp_oos = dp[n_train - 1:]

        cov_train = np.cov(dp_train[1:], dp_train[:-1])[0, 1]
        cov_oos = np.cov(dp_oos[1:], dp_oos[:-1])[0, 1]

        s_price_train = np.sqrt(-cov_train) if cov_train < 0 else 0.0
        s_price_oos = np.sqrt(-cov_oos) if cov_oos < 0 else 0.0

        avg_price_train = close[:n_train].mean()
        avg_price_oos = close[n_train:].mean()

        s_raw_train = s_price_train / avg_price_train if avg_price_train > 0 else 0.0
        s_raw_oos = s_price_oos / avg_price_oos if avg_price_oos > 0 else 0.0

        raw_estimates[coin] = {
            'cov_train': cov_train,
            's_raw_train': s_raw_train,
            's_raw_oos': s_raw_oos,
            's_raw_bps_train': s_raw_train * 10_000,
        }

    # Step 2: Calibrate to Binance average cost of 10 bps
    # Scale factor = target_avg / raw_avg
    raw_values = [raw_estimates[c]['s_raw_train'] for c in COINS]
    raw_avg = np.mean(raw_values) if np.mean(raw_values) > 0 else 1e-8
    scale_factor = BINANCE_AVG_COST / raw_avg

    for coin in COINS:
        s_calibrated = raw_estimates[coin]['s_raw_train'] * scale_factor
        s_calibrated = max(s_calibrated, 0.0005)  # Floor: 5 bps minimum

        s_oos_calibrated = raw_estimates[coin]['s_raw_oos'] * scale_factor
        s_oos_calibrated = max(s_oos_calibrated, 0.0005)

        results[coin] = {
            'cov_train': raw_estimates[coin]['cov_train'],
            's_raw_train': raw_estimates[coin]['s_raw_train'],
            's_raw_bps_train': raw_estimates[coin]['s_raw_bps_train'],
            's_decimal_train': s_calibrated,
            's_decimal_oos': s_oos_calibrated,
            's_bps_train': s_calibrated * 10_000,
            's_bps_oos': s_oos_calibrated * 10_000,
        }

    return results


def estimate_corwin_schultz_slippage():
    """
    Corwin-Schultz (2012) high-low spread estimator on 6-HOUR bars.

    Uses High and Low prices. The key insight:
    - A single bar's high-low range reflects: volatility + spread
    - A two-bar high-low range reflects: 2×volatility + spread
    - By comparing the two, we isolate the spread component

    We resample to 6-hour bars (matching trading frequency), just like Roll.

    Reference: Corwin & Schultz (2012) "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices", Journal of Finance, 67(2).
    """
    REBAL_FREQ = 6
    results = {}
    k = 3 - 2 * np.sqrt(2)  # constant ≈ 0.1716

    for coin in COINS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet"))

        # Resample to 6-hour bars: take max(high), min(low)
        df_6h = df.resample('6h').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        high = df_6h['high'].values
        low = df_6h['low'].values

        train_mask = df_6h.index <= TRAIN_END
        n_train = train_mask.sum()

        def compute_cs(h, l):
            n = len(h)
            alphas = []
            for t in range(1, n):
                hl_t = np.log(h[t] / l[t])
                hl_tm1 = np.log(h[t-1] / l[t-1])
                beta = hl_t**2 + hl_tm1**2
                h_2bar = max(h[t], h[t-1])
                l_2bar = min(l[t], l[t-1])
                gamma = np.log(h_2bar / l_2bar)**2
                sqrt_beta = np.sqrt(beta)
                alpha = (np.sqrt(2) * sqrt_beta - sqrt_beta) / k - np.sqrt(gamma / k)
                alphas.append(alpha)
            alphas = np.array(alphas)
            valid = alphas[alphas > 0]
            if len(valid) > 0:
                avg_alpha = np.mean(valid)
                s = 2 * (np.exp(avg_alpha) - 1) / (1 + np.exp(avg_alpha))
            else:
                s = 0.0
            return s

        s_train = compute_cs(high[:n_train], low[:n_train])
        s_oos = compute_cs(high[n_train:], low[n_train:])

        results[coin] = {
            's_decimal_train': s_train,
            's_decimal_oos': s_oos,
            's_bps_train': s_train * 10_000,
            's_bps_oos': s_oos * 10_000,
        }

    return results


def print_slippage_comparison(roll_results, cs_results):
    """Print comparison table of both estimators."""
    print("\n  Roll model: raw estimates from hourly Δp, calibrated to Binance ~10bps avg")
    print("  Floor of 5 bps applied per coin")
    print("\n  ┌─────────┬──────────────┬──────────────┬──────────────┐")
    print("  │  Coin   │ Roll raw bps │ Roll+floor   │ Corwin-S bps │")
    print("  ├─────────┼──────────────┼──────────────┼──────────────┤")
    for coin in COINS:
        label = coin.replace('USDT', '').center(7)
        raw_bps = roll_results[coin]['s_raw_bps_train']
        r_bps = roll_results[coin]['s_bps_train']
        cs_bps = cs_results[coin]['s_bps_train']
        floor_flag = " *" if raw_bps < 5.0 else "  "
        print(f"  │ {label} │ {raw_bps:>9.2f} bp │ {r_bps:>8.2f} bp{floor_flag}│ {cs_bps:>9.2f} bp │")
    print("  └─────────┴──────────────┴──────────────┴──────────────┘")
    print("  (* = floor applied; raw Roll estimate was below 5 bps)")

    avg_roll = np.mean([roll_results[c]['s_bps_train'] for c in COINS])
    avg_cs = np.mean([cs_results[c]['s_bps_train'] for c in COINS])
    print(f"\n  Average:  Roll = {avg_roll:.2f} bps  |  Corwin-Schultz = {avg_cs:.2f} bps")

    print(f"\n  Binance reference (for context):")
    print(f"    Spot fee: ~10 bps (0.1%) maker/taker")
    print(f"    Futures fee: ~2–5 bps")
    print(f"    Typical bid-ask spread: ~1–7 bps")
    print(f"    Realistic total per trade: ~10–15 bps")

    print(f"\n  We use the Roll model estimate (with floor) for cost computation")
    print(f"  (as specified by the coursework instructions)")

    print(f"\n  Out-of-sample estimates:")
    for coin in COINS:
        label = coin.replace('USDT', '')
        r_oos = roll_results[coin]['s_bps_oos']
        print(f"    {label}: {r_oos:.2f} bps")


# ═══════════════════════════════════════════════════════════════════════
# TASK 4: PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_transaction_costs(theta, returns, slippage_per_coin):
    """
    Compute transaction costs at each time step.

    Cost_t = Σ_i  s_i × |θ_t^i - θ_{t-1}^i × (1 + r_{t-1}^i)|

    In plain English:
    - θ_{t-1}^i × (1 + r_{t-1}^i) = what our old position grew to naturally
    - θ_t^i = what we WANT our position to be now
    - The difference = how much we actually need to TRADE
    - We pay slippage s_i on every dollar of that trade
    """
    n = len(theta)
    costs = np.zeros(n)

    for ci, coin in enumerate(COINS):
        s = slippage_per_coin[coin]
        pos = theta[coin].values
        ret = returns[coin].values

        for t in range(1, n):
            old_pos_grown = pos[t - 1] * (1 + ret[t])
            trade_size = abs(pos[t] - old_pos_grown)
            costs[t] += s * trade_size

    return pd.Series(costs, index=theta.index)


def compute_performance(theta, returns, costs, name, train_end):
    """
    Compute full performance metrics for a strategy.

    Metrics (all computed on NET PnL unless stated):
    - Sharpe  = mean / std × √(24×365)           annualised
    - Sortino = mean / downside_std × √(24×365)   only penalises losses
    - Calmar  = annualised_return / |max_drawdown|
    - Turnover = total traded / (V0 × num_hours)
    - Holding horizon = avg_position / avg_hourly_turnover
    """
    pnl_gross = (theta.shift(1) * returns).sum(axis=1).fillna(0)
    pnl_net = pnl_gross - costs

    pnl_gross_is = pnl_gross.loc[:train_end]
    pnl_gross_oos = pnl_gross.loc[train_end:]
    pnl_net_is = pnl_net.loc[:train_end]
    pnl_net_oos = pnl_net.loc[train_end:]
    costs_is = costs.loc[:train_end]
    costs_oos = costs.loc[train_end:]

    results = {}

    for label, pnl_g, pnl_n, cost_period in [
        ('In-sample', pnl_gross_is, pnl_net_is, costs_is),
        ('Out-of-sample', pnl_gross_oos, pnl_net_oos, costs_oos),
        ('Full period', pnl_gross, pnl_net, costs),
    ]:
        if len(pnl_n) == 0 or pnl_n.std() == 0:
            continue

        sharpe = pnl_n.mean() / pnl_n.std() * np.sqrt(24 * 365)

        downside = pnl_n[pnl_n < 0]
        downside_std = downside.std() if len(downside) > 0 else pnl_n.std()
        sortino = pnl_n.mean() / downside_std * np.sqrt(24 * 365) if downside_std > 0 else 0

        cum_pnl = pnl_n.cumsum()
        drawdown = cum_pnl - cum_pnl.cummax()
        max_dd = drawdown.min()
        hours = len(pnl_n)
        years = hours / (24 * 365)
        total_return = cum_pnl.iloc[-1]
        ann_return = total_return / years if years > 0 else 0
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0

        # Turnover
        theta_period = theta.loc[pnl_n.index]
        returns_period = returns.loc[pnl_n.index]
        total_traded = 0
        for coin in COINS:
            pos = theta_period[coin].values
            ret = returns_period[coin].values
            for t in range(1, len(pos)):
                trade = abs(pos[t] - pos[t-1] * (1 + ret[t]))
                if not np.isnan(trade):
                    total_traded += trade

        avg_hourly_turnover = total_traded / hours if hours > 0 else 0
        turnover_ratio = total_traded / (V0 * hours) if hours > 0 else 0

        avg_position = theta_period.abs().sum(axis=1).mean()
        holding_horizon = avg_position / avg_hourly_turnover if avg_hourly_turnover > 0 else 0

        sharpe_gross = pnl_g.mean() / pnl_g.std() * np.sqrt(24 * 365) if pnl_g.std() > 0 else 0

        results[label] = {
            'sharpe_gross': sharpe_gross,
            'sharpe_net': sharpe,
            'sortino_net': sortino,
            'calmar_net': calmar,
            'total_pnl_gross': pnl_g.sum(),
            'total_pnl_net': pnl_n.sum(),
            'total_costs': cost_period.sum(),
            'return_pct': (pnl_n.sum() / V0) * 100,
            'max_drawdown': max_dd,
            'turnover_ratio': turnover_ratio,
            'holding_horizon_hrs': holding_horizon,
            'num_hours': hours,
        }

    return results, pnl_gross, pnl_net, costs


def print_performance(results, name):
    """Print performance metrics in a clear table."""
    print(f"\n  {'═' * 65}")
    print(f"  {name}")
    print(f"  {'═' * 65}")

    for label in ['In-sample', 'Out-of-sample', 'Full period']:
        if label not in results:
            continue
        r = results[label]
        print(f"\n  {label}:")
        print(f"    Gross PnL:       ${r['total_pnl_gross']:>12,.0f}")
        print(f"    Transaction Costs: ${r['total_costs']:>12,.0f}")
        print(f"    Net PnL:         ${r['total_pnl_net']:>12,.0f}  "
              f"({r['return_pct']:+.1f}% return on $10k)")
        print(f"    Max Drawdown:    ${r['max_drawdown']:>12,.0f}")
        print(f"    ─────────────────────────────────────")
        print(f"    Sharpe  (gross): {r['sharpe_gross']:>8.3f}")
        print(f"    Sharpe  (net):   {r['sharpe_net']:>8.3f}")
        print(f"    Sortino (net):   {r['sortino_net']:>8.3f}")
        print(f"    Calmar  (net):   {r['calmar_net']:>8.3f}")
        print(f"    ─────────────────────────────────────")
        print(f"    Turnover ratio:  {r['turnover_ratio']:>8.4f} per hour")
        print(f"    Holding horizon: {r['holding_horizon_hrs']:>8.1f} hours")


# ═══════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def slippage_sensitivity(theta, returns, base_slippage, name, train_end):
    """
    Test how strategy performance changes with different slippage levels.

    We test multiples of the Corwin-Schultz estimate: [0x, 0.5x, 1x, 2x, 3x, 5x]

    Why this matters:
    - Our estimate might be too low (large orders have more market impact)
    - Or too high (Binance has tight spreads for small orders)
    - If the strategy survives at 3x slippage → it's robust
    - If it dies at 0.5x slippage → it's fragile
    """
    multipliers = [0, 0.5, 1.0, 2.0, 3.0, 5.0]
    sensitivity = []

    for mult in multipliers:
        scaled = {coin: base_slippage[coin] * mult for coin in COINS}
        costs = compute_transaction_costs(theta, returns, scaled)
        pnl_net = (theta.shift(1) * returns).sum(axis=1).fillna(0) - costs

        pnl_oos = pnl_net.loc[train_end:]
        if len(pnl_oos) > 0 and pnl_oos.std() > 0:
            sharpe = pnl_oos.mean() / pnl_oos.std() * np.sqrt(24 * 365)
            total = pnl_oos.sum()
        else:
            sharpe = 0
            total = 0

        sensitivity.append({
            'multiplier': mult,
            'label': f'{mult:.1f}x',
            'oos_sharpe': round(sharpe, 3),
            'oos_pnl': round(total, 0),
            'total_costs_oos': round(costs.loc[train_end:].sum(), 0),
        })

    return pd.DataFrame(sensitivity)


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_performance(pnl_gross_s1, pnl_net_s1, costs_s1,
                     pnl_gross_s2, pnl_net_s2, costs_s2,
                     train_end):
    """Plot cumulative PnL (gross vs net) and drawdowns."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Task 3-4: Transaction Costs & Performance',
                 fontsize=14, fontweight='bold')

    strategies = [
        ('Strategy 1: Mean-Reversion + MVO', pnl_gross_s1, pnl_net_s1, costs_s1),
        ('Strategy 2: Trend-Following', pnl_gross_s2, pnl_net_s2, costs_s2),
    ]

    for col, (sname, pnl_g, pnl_n, cost) in enumerate(strategies):
        ax = axes[0, col]
        cum_gross = pnl_g.cumsum()
        cum_net = pnl_n.cumsum()
        ax.plot(cum_gross.index, cum_gross, linewidth=1, color='blue',
                label='Gross (before costs)', alpha=0.7)
        ax.plot(cum_net.index, cum_net, linewidth=1, color='darkgreen',
                label='Net (after costs)')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7, label='Train/Test')
        ax.fill_between(cum_net.index, cum_gross, cum_net, alpha=0.15, color='red',
                        label='Cost impact')
        ax.set_ylabel('Cumulative PnL (USDT)')
        ax.set_title(f'{sname}\nGross vs Net PnL')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        dd = cum_net - cum_net.cummax()
        ax.fill_between(dd.index, 0, dd, color='red', alpha=0.4)
        ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Drawdown (USDT)')
        ax.set_title('Drawdown (Net of Costs)')
        ax.grid(True, alpha=0.3)

        ax = axes[2, col]
        cum_cost = cost.cumsum()
        ax.plot(cum_cost.index, cum_cost, linewidth=1, color='red')
        ax.axvline(x=train_end, color='red', linestyle='--', alpha=0.7)
        ax.fill_between(cum_cost.index, 0, cum_cost, alpha=0.15, color='red')
        ax.set_ylabel('Cumulative Costs (USDT)')
        ax.set_xlabel('Date')
        ax.set_title('Cumulative Transaction Costs')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig7_performance.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig7_performance.png")
    plt.close()


def plot_sensitivity(sens_s1, sens_s2):
    """Plot slippage sensitivity analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Slippage Sensitivity Analysis (Out-of-Sample)',
                 fontsize=13, fontweight='bold')

    for ax, sens, name in [(axes[0], sens_s1, 'Strategy 1: MR+MVO'),
                            (axes[1], sens_s2, 'Strategy 2: Trend-Following')]:
        x = range(len(sens))
        colors = ['green' if s > 0 else 'red' for s in sens['oos_sharpe']]
        ax.bar(x, sens['oos_sharpe'], color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Slippage Multiplier (× Corwin-Schultz estimate)')
        ax.set_ylabel('Out-of-Sample Sharpe Ratio')
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(sens['label'])
        ax.grid(True, alpha=0.3, axis='y')

        for i, s in enumerate(sens['oos_sharpe']):
            ax.text(i, s + 0.02 if s >= 0 else s - 0.06,
                    f'{s:.2f}', ha='center', va='bottom' if s >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig8_sensitivity.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig8_sensitivity.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  COMP0051 - TASK 3 & 4: TRANSACTION COSTS & PERFORMANCE")
    print("═" * 60)

    # ── Load data ──
    print("\n  Loading data...")
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

    theta_s1 = pd.read_parquet(os.path.join(DATA_DIR, "strategy1_positions.parquet"))
    theta_s2 = pd.read_parquet(os.path.join(DATA_DIR, "strategy2_positions.parquet"))

    common_idx = prices.index.intersection(returns.index) \
                              .intersection(theta_s1.index) \
                              .intersection(theta_s2.index)
    prices = prices.loc[common_idx]
    returns = returns.loc[common_idx]
    theta_s1 = theta_s1.loc[common_idx]
    theta_s2 = theta_s2.loc[common_idx]

    print(f"  Loaded: {len(prices)} bars, {len(COINS)} coins")
    print(f"  Coins: {', '.join(c.replace('USDT','') for c in COINS)}")

    # ══════════════════════════════════════════════════════════════════
    # TASK 3: SLIPPAGE ESTIMATION — TWO METHODS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TASK 3: TRANSACTION COSTS")
    print("=" * 60)

    # Method 1: Roll (1984)
    print("\n  Method 1: Roll (1984) model")
    print("  Formula: s = √( -Cov(Δp_t, Δp_{t-1}) )")
    roll_results = estimate_roll_slippage()

    # Method 2: Corwin-Schultz (2012)
    print("\n  Method 2: Corwin-Schultz (2012) high-low estimator")
    print("  Uses bar High/Low prices to separate volatility from spread")
    cs_results = estimate_corwin_schultz_slippage()

    # Compare the two
    print_slippage_comparison(roll_results, cs_results)

    # Use Roll model for main computation (as specified in coursework)
    slippage_per_coin = {coin: roll_results[coin]['s_decimal_train'] for coin in COINS}

    print(f"\n  Using calibrated Roll model estimates for cost computation:")
    for coin in COINS:
        s = slippage_per_coin[coin]
        print(f"    {coin.replace('USDT','')}: {s:.6f} ({s*10000:.2f} bps)")
    avg_s = np.mean(list(slippage_per_coin.values()))
    print(f"    Average: {avg_s:.6f} ({avg_s*10000:.2f} bps)")
    print(f"    For every $10,000 traded, you pay ~${avg_s * 10000:.2f} in slippage")

    # ══════════════════════════════════════════════════════════════════
    # TASK 4: PERFORMANCE (NET OF COSTS)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TASK 4: PERFORMANCE METRICS")
    print("=" * 60)

    print("\n  Computing transaction costs for Strategy 1...")
    costs_s1 = compute_transaction_costs(theta_s1, returns, slippage_per_coin)
    results_s1, pnl_g_s1, pnl_n_s1, _ = compute_performance(
        theta_s1, returns, costs_s1, "Strategy 1", TRAIN_END)
    print_performance(results_s1, "Strategy 1: Multi-Asset Mean-Reversion + MVO")

    print("\n  Computing transaction costs for Strategy 2...")
    costs_s2 = compute_transaction_costs(theta_s2, returns, slippage_per_coin)
    results_s2, pnl_g_s2, pnl_n_s2, _ = compute_performance(
        theta_s2, returns, costs_s2, "Strategy 2", TRAIN_END)
    print_performance(results_s2, "Strategy 2: Trend-Following")

    # ══════════════════════════════════════════════════════════════════
    # SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SLIPPAGE SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\n  Testing: what happens if real slippage is 2x, 3x, 5x our estimate?")
    sens_s1 = slippage_sensitivity(theta_s1, returns, slippage_per_coin, "S1", TRAIN_END)
    sens_s2 = slippage_sensitivity(theta_s2, returns, slippage_per_coin, "S2", TRAIN_END)

    print("\n  Strategy 1 sensitivity (OOS):")
    print(sens_s1.to_string(index=False))
    print("\n  Strategy 2 sensitivity (OOS):")
    print(sens_s2.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # REINVESTMENT LOGIC
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  REINVESTMENT LOGIC")
    print("=" * 60)
    print("""
  We use a FIXED $100,000 gross exposure cap (not reinvesting profits).

  Why? Three reasons:
  1. The coursework specifies Σ|θ_t^i| ≤ $100,000 as a hard constraint.
     This is a regulatory-style limit, not a portfolio-size target.
  2. Reinvesting profits would compound risk — a strategy that made $50k
     profit would push capital to $60k, enabling $600k exposure (60×10x).
     This would violate the $100k constraint.
  3. Unallocated capital (profits + unused margin) is held as USDT cash,
     acting as a buffer against drawdowns.
  """)

    for sname, results in [("Strategy 1", results_s1), ("Strategy 2", results_s2)]:
        r = results['Full period']
        end_capital = V0 + r['total_pnl_net']
        print(f"  {sname}: $10,000 → ${end_capital:,.0f} "
              f"({r['return_pct']:+.1f}% total return)")

    # ══════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════
    print("\n  Generating plots...")
    plot_performance(pnl_g_s1, pnl_n_s1, costs_s1,
                     pnl_g_s2, pnl_n_s2, costs_s2, TRAIN_END)
    plot_sensitivity(sens_s1, sens_s2)

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    summary_rows = []
    for sname, results in [("Strategy 1 (MR+MVO)", results_s1),
                           ("Strategy 2 (Trend)", results_s2)]:
        for period, r in results.items():
            row = {'strategy': sname, 'period': period}
            row.update(r)
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(DATA_DIR, "performance_summary.csv"), index=False)
    print("  ✅ Saved performance_summary.csv")

    # Save both slippage estimates
    roll_df = pd.DataFrame(roll_results).T
    roll_df.to_csv(os.path.join(DATA_DIR, "roll_slippage.csv"))
    cs_df = pd.DataFrame(cs_results).T
    cs_df.to_csv(os.path.join(DATA_DIR, "cs_slippage.csv"))
    print("  ✅ Saved roll_slippage.csv and cs_slippage.csv")

    sens_s1.to_csv(os.path.join(DATA_DIR, "sensitivity_s1.csv"), index=False)
    sens_s2.to_csv(os.path.join(DATA_DIR, "sensitivity_s2.csv"), index=False)
    print("  ✅ Saved sensitivity CSVs")

    print("\n" + "=" * 60)
    print("  TASKS 3 & 4 COMPLETE!")
    print("  → Proceed to Task 5 (Next Steps)")
    print("=" * 60)
