"""
COMP0051 - Algorithmic Trading Coursework
Task 5

Analysis to support the discussion of:
  1. Performance across different time horizons
  2. Market regime sensitivity (bull / bear / sideways)
  3. Live trading viability (leverage, drawdowns, latency)
  4. Strategy improvements (regime-aware allocation, combination, etc.)

All methods grounded in the course reading list:
  - Cont (2001): volatility clustering → non-stationary Sharpe
  - Dacco & Satchell (1999): regime-switching models
  - Moskowitz, Ooi & Pedersen (2012): TSMOM profits vary by horizon/regime
  - Avellaneda & Lee (2010): mean-reversion works in range-bound markets
  - Cartea et al. (2015) Ch. 7: execution risk, dynamic rebalancing
  - Markowitz (1952), Bouchaud & Potters (2003): diversification benefit
  - Tsay (2010) Ch. 11: adaptive/online parameter methods
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

# Calibrated Roll slippage (from Task 3)
SLIPPAGE = {
    'BTCUSDT': 0.000500,   # 5 bps
    'ETHUSDT': 0.000500,   # 5 bps
    'DOGEUSDT': 0.001614,  # 16 bps
    'BNBUSDT': 0.000691,   # 7 bps
    'XRPUSDT': 0.001770,   # 18 bps
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load prices, returns, and positions."""
    prices = pd.DataFrame()
    returns = pd.DataFrame()
    for coin in COINS:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet"))
        dr = pd.read_parquet(os.path.join(DATA_DIR, f"{coin}_returns.parquet"))
        prices[coin] = df['close']
        returns[coin] = dr['excess_return']

    theta_s1 = pd.read_parquet(os.path.join(DATA_DIR, "strategy1_positions.parquet"))
    theta_s2 = pd.read_parquet(os.path.join(DATA_DIR, "strategy2_positions.parquet"))

    idx = prices.index.intersection(returns.index) \
                      .intersection(theta_s1.index) \
                      .intersection(theta_s2.index)
    prices = prices.loc[idx]
    returns = returns.loc[idx]
    theta_s1 = theta_s1.loc[idx]
    theta_s2 = theta_s2.loc[idx]

    return prices, returns, theta_s1, theta_s2


def compute_costs(theta, returns):
    """Transaction costs using calibrated Roll slippage."""
    n = len(theta)
    costs = np.zeros(n)
    for coin in COINS:
        s = SLIPPAGE[coin]
        pos = theta[coin].values
        ret = returns[coin].values
        for t in range(1, n):
            trade = abs(pos[t] - pos[t - 1] * (1 + ret[t]))
            if not np.isnan(trade):
                costs[t] += s * trade
    return pd.Series(costs, index=theta.index)


def compute_net_pnl(theta, returns):
    """Gross PnL, costs, and net PnL series."""
    pnl_gross = (theta.shift(1) * returns).sum(axis=1).fillna(0)
    costs = compute_costs(theta, returns)
    pnl_net = pnl_gross - costs
    return pnl_gross, costs, pnl_net


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: ROLLING SHARPE ACROSS TIME HORIZONS
# ═══════════════════════════════════════════════════════════════════════

def rolling_sharpe_analysis(pnl_s1, pnl_s2, train_end):
    """Compute rolling Sharpe at 1m, 3m, 6m windows."""
    windows = {
        '1-month (720h)': 720,
        '3-month (2160h)': 2160,
        '6-month (4320h)': 4320,
    }

    oos_s1 = pnl_s1.loc[train_end:]
    oos_s2 = pnl_s2.loc[train_end:]
    annualise = np.sqrt(24 * 365)

    results = {}
    for sname, oos in [('S1', oos_s1), ('S2', oos_s2)]:
        results[sname] = {}
        for wname, w in windows.items():
            r_mean = oos.rolling(w).mean()
            r_std = oos.rolling(w).std()
            r_sharpe = (r_mean / r_std * annualise).dropna()
            results[sname][wname] = r_sharpe

            pct_positive = (r_sharpe > 0).mean() * 100
            print(f"    {sname} {wname}: mean={r_sharpe.mean():.2f}, "
                  f"min={r_sharpe.min():.2f}, max={r_sharpe.max():.2f}, "
                  f"positive={pct_positive:.0f}% of time")

    return results, windows


def plot_rolling_sharpe(results, windows, train_end):
    """Figure 10: Rolling Sharpe across time horizons."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Rolling Sharpe Ratio Across Time Horizons\n(Out-of-Sample, Net of Costs)',
                 fontsize=13, fontweight='bold')

    colours = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, sname, title in [(axes[0], 'S1', 'Strategy 1: Mean-Reversion + MVO'),
                              (axes[1], 'S2', 'Strategy 2: Trend-Following')]:
        for (wname, series), colour in zip(results[sname].items(), colours):
            ax.plot(series.index, series.values, linewidth=0.8,
                    color=colour, alpha=0.9, label=wname)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle=':', alpha=0.4, label='Sharpe = 1')
        ax.axhline(y=-1, color='red', linestyle=':', alpha=0.4, label='Sharpe = -1')
        ax.set_ylabel('Annualised Sharpe')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-4, 6)

    axes[1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig10_rolling_sharpe.png'),
                dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig10_rolling_sharpe.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: MARKET REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def regime_analysis(prices, pnl_s1, pnl_s2, theta_s1, theta_s2, train_end):
    """
    Classify market into bull/bear/sideways regimes and measure
    strategy performance in each.

    Regime definition (using 168h = 1-week rolling return of
    equal-weighted crypto index):
      Bull:     weekly return > +5%  (strong uptrend)
      Bear:     weekly return < -5%  (strong downtrend)
      Sideways: in between           (range-bound / choppy)

    The ±5% threshold ≈ 1 standard deviation of weekly crypto returns
    (BTC annualised vol ~50%, so weekly vol ≈ 50/√52 ≈ 7%).
    """
    # Equal-weighted market return
    mkt_ret = prices[COINS].pct_change().mean(axis=1)
    weekly_ret = mkt_ret.rolling(168).sum()  # approximate 1-week cumulative return

    # Classify regimes
    regime = pd.Series('Sideways', index=prices.index)
    regime[weekly_ret > 0.05] = 'Bull'
    regime[weekly_ret < -0.05] = 'Bear'

    # OOS only
    oos_mask = prices.index > train_end
    regime_oos = regime[oos_mask]
    pnl_s1_oos = pnl_s1[oos_mask]
    pnl_s2_oos = pnl_s2[oos_mask]

    annualise = np.sqrt(24 * 365)
    regime_stats = []

    for reg in ['Bull', 'Bear', 'Sideways']:
        mask = regime_oos == reg
        hours = mask.sum()
        pct = hours / len(regime_oos) * 100

        for sname, pnl in [('S1', pnl_s1_oos), ('S2', pnl_s2_oos)]:
            p = pnl[mask]
            if len(p) > 0 and p.std() > 0:
                sharpe = p.mean() / p.std() * annualise
                total = p.sum()
            else:
                sharpe = 0
                total = 0

            regime_stats.append({
                'regime': reg, 'strategy': sname,
                'hours': hours, 'pct_oos': round(pct, 1),
                'sharpe': round(sharpe, 3),
                'pnl': round(total, 0),
            })

    stats_df = pd.DataFrame(regime_stats)

    # Print table
    print("\n  ┌──────────┬──────────┬────────┬──────────┬──────────────┐")
    print("  │ Regime   │ Strategy │ Hours  │ Sharpe   │ Net PnL      │")
    print("  ├──────────┼──────────┼────────┼──────────┼──────────────┤")
    for _, r in stats_df.iterrows():
        print(f"  │ {r['regime']:>8s} │   {r['strategy']}     │ {r['hours']:>5d}  │"
              f" {r['sharpe']:>7.3f}  │ ${r['pnl']:>11,.0f} │")
    print("  └──────────┴──────────┴────────┴──────────┴──────────────┘")
    print(f"\n  Regime distribution (OOS): "
          f"Bull={stats_df[stats_df['regime']=='Bull']['pct_oos'].iloc[0]:.0f}%, "
          f"Bear={stats_df[stats_df['regime']=='Bear']['pct_oos'].iloc[0]:.0f}%, "
          f"Sideways={stats_df[stats_df['regime']=='Sideways']['pct_oos'].iloc[0]:.0f}%")

    return stats_df, regime_oos


def plot_regime_analysis(stats_df, pnl_s1, pnl_s2, regime_oos, train_end):
    """Figure 11: Regime-conditional performance."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Strategy Performance by Market Regime (Out-of-Sample)',
                 fontsize=13, fontweight='bold')

    regimes = ['Bull', 'Bear', 'Sideways']
    reg_colors = {'Bull': '#4CAF50', 'Bear': '#F44336', 'Sideways': '#9E9E9E'}

    # ── Top-left: Sharpe by regime ──
    ax = axes[0, 0]
    x = np.arange(len(regimes))
    w = 0.35
    s1_sharpes = [stats_df[(stats_df['regime'] == r) & (stats_df['strategy'] == 'S1')]['sharpe'].values[0]
                  for r in regimes]
    s2_sharpes = [stats_df[(stats_df['regime'] == r) & (stats_df['strategy'] == 'S2')]['sharpe'].values[0]
                  for r in regimes]
    bars1 = ax.bar(x - w/2, s1_sharpes, w, label='S1 (MR+MVO)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + w/2, s2_sharpes, w, label='S2 (Trend)', color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Annualised Sharpe')
    ax.set_title('Sharpe Ratio by Regime')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05 if h >= 0 else h - 0.15,
                    f'{h:.2f}', ha='center', fontsize=8)

    # ── Top-right: PnL by regime ──
    ax = axes[0, 1]
    s1_pnls = [stats_df[(stats_df['regime'] == r) & (stats_df['strategy'] == 'S1')]['pnl'].values[0]
               for r in regimes]
    s2_pnls = [stats_df[(stats_df['regime'] == r) & (stats_df['strategy'] == 'S2')]['pnl'].values[0]
               for r in regimes]
    ax.bar(x - w/2, s1_pnls, w, label='S1 (MR+MVO)', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, s2_pnls, w, label='S2 (Trend)', color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Net PnL (USDT)')
    ax.set_title('Net PnL by Regime')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Bottom: Cumulative PnL coloured by regime ──
    oos_start = train_end
    for col, (sname, pnl) in enumerate([('Strategy 1: MR+MVO', pnl_s1),
                                         ('Strategy 2: Trend-Following', pnl_s2)]):
        ax = axes[1, col]
        pnl_oos = pnl.loc[oos_start:]
        cum = pnl_oos.cumsum()

        # Plot segments coloured by regime
        for reg in regimes:
            mask = (regime_oos == reg).reindex(cum.index, fill_value=False)
            reg_cum = cum.copy()
            reg_cum[~mask] = np.nan
            ax.plot(reg_cum.index, reg_cum.values, linewidth=1.5,
                    color=reg_colors[reg], alpha=0.8, label=reg)

        # Also plot full line faintly
        ax.plot(cum.index, cum.values, linewidth=0.3, color='black', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Cumulative Net PnL (USDT)')
        ax.set_xlabel('Date')
        ax.set_title(f'{sname}\nCumulative PnL Coloured by Regime')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig11_regime_analysis.png'),
                dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig11_regime_analysis.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: STRATEGY COMBINATION & DIVERSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def combination_analysis(pnl_s1, pnl_s2, train_end):
    """Analyse diversification benefit of combining S1 and S2."""
    oos_s1 = pnl_s1.loc[train_end:]
    oos_s2 = pnl_s2.loc[train_end:]
    annualise = np.sqrt(24 * 365)

    # Correlation
    corr = oos_s1.corr(oos_s2)
    print(f"    PnL correlation (S1 vs S2): {corr:.3f}")

    # 50/50 combination
    pnl_combo = 0.5 * oos_s1 + 0.5 * oos_s2

    strategies = [
        ('S1 alone', oos_s1),
        ('S2 alone', oos_s2),
        ('50/50 combo', pnl_combo),
    ]

    print(f"\n    {'Strategy':<15s} {'Sharpe':>8s} {'Net PnL':>10s} {'MaxDD':>10s} {'Sortino':>8s}")
    print(f"    {'─'*55}")

    for name, pnl in strategies:
        sharpe = pnl.mean() / pnl.std() * annualise if pnl.std() > 0 else 0
        total = pnl.sum()
        dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        downside = pnl[pnl < 0]
        sortino = pnl.mean() / downside.std() * annualise if len(downside) > 0 and downside.std() > 0 else 0
        print(f"    {name:<15s} {sharpe:>8.3f} ${total:>9,.0f} ${dd:>9,.0f} {sortino:>8.3f}")

    return corr


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: LIVE TRADING VIABILITY
# ═══════════════════════════════════════════════════════════════════════

def live_trading_analysis(theta_s1, theta_s2, pnl_s1, pnl_s2, train_end):
    """Analyse leverage, drawdown, and live trading feasibility."""
    oos_mask = theta_s1.index > train_end

    for sname, theta, pnl in [('S1', theta_s1, pnl_s1), ('S2', theta_s2, pnl_s2)]:
        gross = theta[oos_mask].abs().sum(axis=1)
        lev = gross / V0

        pnl_oos = pnl[oos_mask]
        cum = pnl_oos.cumsum()
        dd = cum - cum.cummax()

        # Drawdown duration
        underwater = dd < 0
        if underwater.any():
            # Find longest streak of underwater hours
            groups = (~underwater).cumsum()
            uw_groups = underwater.groupby(groups)
            max_dd_duration = uw_groups.sum().max()
            pct_underwater = underwater.mean() * 100
        else:
            max_dd_duration = 0
            pct_underwater = 0

        print(f"\n    {sname}:")
        print(f"      Leverage: mean={lev.mean():.1f}x, median={lev.median():.1f}x, "
              f"p95={lev.quantile(0.95):.1f}x, max={lev.max():.1f}x")
        print(f"      Max drawdown: ${dd.min():,.0f}")
        print(f"      Max DD duration: {max_dd_duration:,.0f} hours "
              f"({max_dd_duration/24:.0f} days)")
        print(f"      Time underwater: {pct_underwater:.1f}% of OOS period")
        print(f"      Margin requirement: ${lev.mean() * V0:,.0f} avg gross exposure")

        # Would the strategy survive without additional capital?
        if abs(dd.min()) > V0:
            print(f"      ⚠️  Max drawdown (${abs(dd.min()):,.0f}) exceeds V0 (${V0:,})")
            print(f"         → Would need additional capital buffer in live trading")
        else:
            print(f"      ✅  Max drawdown within V0 — survivable without extra capital")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: IMPROVEMENT DISCUSSION SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_discussion_summary():
    """Print structured discussion points for the report."""
    print("""
  ═══════════════════════════════════════════════════════════════
  DISCUSSION POINTS FOR REPORT (Task 5)
  ═══════════════════════════════════════════════════════════════

  1. PERFORMANCE ACROSS TIME HORIZONS
     • Rolling Sharpe shows whether alpha is persistent or episodic
     • Shorter windows (1-month) → more volatile, reveals regime sensitivity
     • Longer windows (6-month) → smoother, shows underlying trend
     • Reference: Moskowitz et al. (2012) document that TSMOM profits
       vary across 1m, 3m, 12m horizons

  2. MARKET REGIME SENSITIVITY
     • S1 (mean-reversion) → expected to perform well in sideways/choppy
       markets, poorly in strong trends (momentum crash risk)
       Reference: Avellaneda & Lee (2010)
     • S2 (trend-following) → expected to perform well in trending
       markets, poorly in reversals/whipsaws
       Reference: Moskowitz, Ooi & Pedersen (2012)
     • Complementarity: S1 and S2 may hedge each other across regimes
       Reference: Dacco & Satchell (1999) on regime-switching

  3. LIVE TRADING VIABILITY
     • Monthly rebalancing is highly feasible operationally
       (no HFT infrastructure needed)
     • Slippage sensitivity (Task 4): both strategies survive even at
       5× estimated slippage → robust to execution quality
     • Leverage requirements: avg 2.7–4.8× is achievable on Binance
       (up to 20× available for crypto futures)
     • Key risk: drawdown can exceed initial capital ($10k) — need
       additional margin buffer or stop-loss rules
     • Reference: Cartea et al. (2015) Ch. 7 on execution risk

  4. STRATEGY IMPROVEMENTS
     (a) Regime-aware allocation: reduce MR exposure when market is
         trending, reduce TF exposure in choppy conditions
         → Reference: Dacco & Satchell (1999)
     (b) Dynamic rebalancing frequency: rebalance more often when
         signals change rapidly, less often when stable
         → Reference: Cartea et al. (2015) on optimal execution
     (c) Volatility targeting: scale positions inversely with
         realised volatility to stabilise risk contribution
         → Reference: Moskowitz et al. (2012) use vol-scaling
     (d) Expand asset universe: more coins enable better cross-
         sectional diversification for MR signals
         → Reference: Avellaneda & Lee (2010)
     (e) Online parameter adaptation: replace static grid search
         with rolling/expanding window recalibration
         → Reference: Tsay (2010) Ch. 11 on adaptive methods
     (f) Combine S1 + S2: if correlation is low/negative, a portfolio
         of both strategies has higher Sharpe per Markowitz (1952)
         → Reference: Bouchaud & Potters (2003) on portfolio risk
  """)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  COMP0051 - TASK 5: NEXT STEPS")
    print("═" * 60)

    # ── Load data ──
    print("\n  Loading data...")
    prices, returns, theta_s1, theta_s2 = load_data()
    print(f"  Loaded: {len(prices)} bars, {len(COINS)} coins")

    # Compute net PnL
    _, _, pnl_net_s1 = compute_net_pnl(theta_s1, returns)
    _, _, pnl_net_s2 = compute_net_pnl(theta_s2, returns)

    # ══════════════════════════════════════════════════════════════════
    # 1. ROLLING SHARPE ACROSS TIME HORIZONS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  1. PERFORMANCE ACROSS TIME HORIZONS")
    print("=" * 60)
    print("\n  Rolling Sharpe (OOS, net of costs):")
    rs_results, rs_windows = rolling_sharpe_analysis(pnl_net_s1, pnl_net_s2, TRAIN_END)
    plot_rolling_sharpe(rs_results, rs_windows, TRAIN_END)

    # ══════════════════════════════════════════════════════════════════
    # 2. MARKET REGIME ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  2. MARKET REGIME ANALYSIS")
    print("  (Dacco & Satchell 1999; Cont 2001)")
    print("=" * 60)
    regime_stats, regime_oos = regime_analysis(
        prices, pnl_net_s1, pnl_net_s2, theta_s1, theta_s2, TRAIN_END)
    plot_regime_analysis(regime_stats, pnl_net_s1, pnl_net_s2, regime_oos, TRAIN_END)

    # ══════════════════════════════════════════════════════════════════
    # 3. STRATEGY COMBINATION & DIVERSIFICATION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  3. COMBINATION POTENTIAL")
    print("  (Markowitz 1952; Bouchaud & Potters 2003)")
    print("=" * 60)
    corr = combination_analysis(pnl_net_s1, pnl_net_s2, TRAIN_END)

    # ══════════════════════════════════════════════════════════════════
    # 4. LIVE TRADING VIABILITY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  4. LIVE TRADING VIABILITY")
    print("  (Cartea et al. 2015 Ch. 7)")
    print("=" * 60)
    live_trading_analysis(theta_s1, theta_s2, pnl_net_s1, pnl_net_s2, TRAIN_END)

    # ══════════════════════════════════════════════════════════════════
    # 5. DISCUSSION SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print_discussion_summary()

    # ── Save summary CSV ──
    regime_stats.to_csv(os.path.join(DATA_DIR, 'next_steps_regime.csv'), index=False)
    print("  ✅ Saved next_steps_regime.csv")

    print("\n" + "=" * 60)
    print("  TASK 5 COMPLETE!")
    print("  Generated: fig10_rolling_sharpe.png, fig11_regime_analysis.png")
    print("  → Ready for the written report and video presentation")
    print("=" * 60)
