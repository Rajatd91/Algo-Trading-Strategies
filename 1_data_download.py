"""
COMP0051 - Algorithmic Trading Coursework
Task 1: Data Download, Cleaning, and Visualization

Downloads HLOCV data from Binance for BTC, ETH, and altcoins (hourly bars),
cleans the data, computes excess simple returns, and plots return time series.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client

# ─── Configuration ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

COINS = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT']
INTERVAL = Client.KLINE_INTERVAL_1HOUR   # hourly bars
START_DATE = "1 Jan, 2024"
END_DATE = "25 Mar, 2026"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Download HLOCV Data from Binance
# ═══════════════════════════════════════════════════════════════════════

def download_ohlcv(symbol, interval, start_date, end_date):
    """Download OHLCV (HLOCV) data from Binance public API (no key needed)."""
    client = Client()  # No API key needed for historical klines

    print(f"  Downloading {symbol} ({interval}) from {start_date} to {end_date}...")

    klines = client.get_historical_klines(
        symbol, interval, start_date, end_date
    )

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Keep only HLOCV columns
    df = df[['timestamp', 'high', 'low', 'open', 'close', 'volume']]

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Convert prices and volume to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df


def download_all_coins():
    """Download data for all coins and save as parquet."""
    print("=" * 60)
    print("STEP 1: Downloading HLOCV data from Binance")
    print("=" * 60)

    all_data = {}
    for coin in COINS:
        df = download_ohlcv(coin, INTERVAL, START_DATE, END_DATE)

        # Save raw data
        raw_path = os.path.join(DATA_DIR, f"{coin}_1h_raw.parquet")
        df.to_parquet(raw_path)

        all_data[coin] = df
        print(f"  ✅ {coin}: {len(df)} rows | {df.index[0]} → {df.index[-1]}")

    print()
    return all_data


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Clean Data and Repair Outliers
# ═══════════════════════════════════════════════════════════════════════

def clean_data(df, symbol):
    """
    Clean HLOCV data:
    1. Remove duplicate timestamps
    2. Fill missing bars (forward fill)
    3. Detect and repair outliers using rolling median + MAD method
    4. Ensure high >= low, high >= open/close, low <= open/close
    """
    original_len = len(df)

    # 1. Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    dupes_removed = original_len - len(df)

    # 2. Reindex to complete hourly frequency and forward-fill gaps
    full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq='h')
    missing_before = len(full_index) - len(df)
    df = df.reindex(full_index)
    df.index.name = 'timestamp'

    # Forward fill missing bars (market was likely closed or data gap)
    df = df.ffill()
    # If any leading NaNs, backfill them
    df = df.bfill()

    # 3. Detect and repair outliers using rolling median + MAD
    #    An outlier is a close price that deviates > 5 MADs from rolling median
    window = 48  # 48-hour rolling window
    outliers_fixed = 0

    for col in ['open', 'high', 'low', 'close']:
        rolling_median = df[col].rolling(window=window, center=True, min_periods=10).median()
        rolling_mad = (df[col] - rolling_median).abs().rolling(window=window, center=True, min_periods=10).median()

        # Avoid division by zero
        rolling_mad = rolling_mad.replace(0, np.nan).ffill().bfill()

        # Flag outliers: > 5 MADs from median
        z_scores = (df[col] - rolling_median).abs() / rolling_mad
        outlier_mask = z_scores > 5

        # Replace outliers with rolling median
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            df.loc[outlier_mask, col] = rolling_median[outlier_mask]
            outliers_fixed += n_outliers

    # 4. Ensure HLOCV consistency
    #    high should be >= open, close, low; low should be <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    # Volume should be non-negative
    df['volume'] = df['volume'].clip(lower=0)

    print(f"  {symbol}: duplicates removed={dupes_removed}, "
          f"gaps filled={missing_before}, outliers fixed={outliers_fixed}")

    return df


def clean_all_data(all_data):
    """Clean data for all coins and save cleaned versions."""
    print("=" * 60)
    print("STEP 2: Cleaning data and repairing outliers")
    print("=" * 60)

    cleaned = {}
    for coin, df in all_data.items():
        df_clean = clean_data(df.copy(), coin)

        # Save cleaned data
        clean_path = os.path.join(DATA_DIR, f"{coin}_1h_clean.parquet")
        df_clean.to_parquet(clean_path)

        cleaned[coin] = df_clean

    print()
    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Download Risk-Free Rate and Compute Excess Returns
# ═══════════════════════════════════════════════════════════════════════

def get_risk_free_rate(start_date, end_date):
    """
    Download Effective Federal Funds Rate from FRED.
    Uses the FRED CSV download endpoint (no API key needed).

    The Fed Funds rate is an annualised rate. We convert to hourly:
        r_hourly = (1 + r_annual)^(1/(365*24)) - 1
    """
    print("  Downloading Effective Fed Funds Rate from FRED...")

    # FRED public CSV endpoint
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=DFF&cosd={start_date}&coed={end_date}"
    )

    try:
        rf = pd.read_csv(url)
        rf['observation_date'] = pd.to_datetime(rf['observation_date'])
        rf.set_index('observation_date', inplace=True)
        rf.columns = ['rate_annual_pct']
        # FRED may return '.' for missing values — convert to NaN and forward fill
        rf['rate_annual_pct'] = pd.to_numeric(rf['rate_annual_pct'], errors='coerce')
        rf = rf.ffill().bfill()

        # Convert from percentage to decimal
        rf['rate_annual'] = rf['rate_annual_pct'] / 100.0

        # Convert annual rate to hourly rate
        # r_hourly = (1 + r_annual)^(1/(365*24)) - 1
        rf['rate_hourly'] = (1 + rf['rate_annual']) ** (1 / (365 * 24)) - 1

        # Reindex to hourly frequency (forward fill daily rate to each hour)
        hourly_index = pd.date_range(
            start=rf.index[0], end=rf.index[-1], freq='h'
        )
        rf = rf.reindex(hourly_index).ffill().bfill()
        rf.index.name = 'timestamp'

        # Save
        rf.to_parquet(os.path.join(DATA_DIR, "risk_free_rate.parquet"))

        avg_annual = rf['rate_annual'].mean() * 100
        avg_hourly = rf['rate_hourly'].mean()
        print(f"  ✅ Risk-free rate: avg annual={avg_annual:.2f}%, "
              f"avg hourly={avg_hourly:.8f}")
        print(f"     Date range: {rf.index[0]} → {rf.index[-1]}")
        print(f"     Rows: {len(rf)}")

        return rf

    except Exception as e:
        print(f"  ⚠️ Could not download from FRED: {e}")
        print("  → Using constant Fed Funds rate of 5.25% (approximate 2023-2024 avg)")

        # Fallback: create a constant risk-free rate
        hourly_index = pd.date_range(start=start_date, end=end_date, freq='h')
        annual_rate = 0.0525
        hourly_rate = (1 + annual_rate) ** (1 / (365 * 24)) - 1

        rf = pd.DataFrame({
            'rate_annual_pct': annual_rate * 100,
            'rate_annual': annual_rate,
            'rate_hourly': hourly_rate
        }, index=hourly_index)
        rf.index.name = 'timestamp'

        rf.to_parquet(os.path.join(DATA_DIR, "risk_free_rate.parquet"))
        return rf


def compute_excess_returns(cleaned_data, rf):
    """
    Compute excess simple returns for each coin.

    r_t^e = (p_t - p_{t-1}) / p_{t-1}  -  r_{t-1}^f

    where r_{t-1}^f is the risk-free rate at the PREVIOUS time step.
    """
    print("  Computing excess returns for each coin...")

    returns_dict = {}

    for coin, df in cleaned_data.items():
        # Simple return
        simple_return = df['close'].pct_change()

        # Align risk-free rate with coin's index
        rf_aligned = rf['rate_hourly'].reindex(df.index).ffill().bfill()

        # Excess return = simple return - lagged risk-free rate
        # r_t^e = (p_t - p_{t-1})/p_{t-1} - r_{t-1}^f
        excess_return = simple_return - rf_aligned.shift(1)

        # Store in a DataFrame
        ret_df = pd.DataFrame({
            'close': df['close'],
            'simple_return': simple_return,
            'risk_free_rate': rf_aligned,
            'excess_return': excess_return
        })

        # Drop the first NaN row
        ret_df = ret_df.dropna()

        # Save
        ret_df.to_parquet(os.path.join(DATA_DIR, f"{coin}_returns.parquet"))

        returns_dict[coin] = ret_df

        avg_ret = ret_df['simple_return'].mean() * 24 * 365 * 100  # annualised %
        avg_rf = ret_df['risk_free_rate'].mean() * 24 * 365 * 100  # annualised %
        print(f"  {coin}: avg annualised return={avg_ret:+.1f}%, "
              f"avg risk-free={avg_rf:.2f}%")

    return returns_dict


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Explanation — Why r^f Can Be Ignored in High-Freq Crypto
# ═══════════════════════════════════════════════════════════════════════

def print_rf_explanation(returns_dict):
    """Print explanation of why r^f can be ignored in mid-to-high frequency crypto."""
    print("=" * 60)
    print("WHY r^f CAN BE IGNORED IN MID-TO-HIGH FREQUENCY CRYPTO")
    print("=" * 60)

    # Pick BTC as example
    btc = returns_dict['BTCUSDT']

    avg_hourly_return = btc['simple_return'].std()  # hourly vol
    avg_hourly_rf = btc['risk_free_rate'].mean()

    ratio = avg_hourly_return / avg_hourly_rf if avg_hourly_rf > 0 else float('inf')

    print(f"""
    The Effective Fed Funds Rate is approximately 5.25% per annum, which
    translates to roughly {avg_hourly_rf:.8f} per hour.

    Meanwhile, BTC hourly volatility (std of returns) is {avg_hourly_return:.6f},
    which is ~{ratio:.0f}x larger than the hourly risk-free rate.

    In mid-to-high frequency crypto trading, r^f can be safely ignored because:

    1. MAGNITUDE: The risk-free rate per bar is negligibly small compared to
       crypto returns. At hourly frequency, r^f ≈ 0.0006% per hour, while
       typical crypto moves are 0.1-1% per hour.

    2. SIGNAL-TO-NOISE: Subtracting r^f from returns has virtually zero impact
       on strategy signals, position sizing, or performance metrics.

    3. HOLDING PERIOD: In high-frequency strategies with holding periods of
       hours to days, the cumulative risk-free return earned/forgone is
       trivially small relative to trading P&L.

    4. CRYPTO HAS NO TRUE RISK-FREE RATE: Unlike equities where you could
       invest in T-bills, crypto markets operate 24/7 and there is no
       equivalent "risk-free" crypto instrument. The Fed Funds rate is a
       fiat-world concept.
    """)


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: Plot Return Time Series
# ═══════════════════════════════════════════════════════════════════════

def plot_returns(returns_dict, cleaned_data):
    """Create publication-quality plots of price and return time series."""
    print("=" * 60)
    print("STEP 5: Plotting return time series")
    print("=" * 60)

    # Color scheme
    colors = {
        'BTCUSDT': '#F7931A',   # Bitcoin orange
        'ETHUSDT': '#627EEA',   # Ethereum blue
        'DOGEUSDT': '#C2A633',  # Doge gold
        'BNBUSDT': '#F3BA2F',   # BNB yellow
        'XRPUSDT': '#00AAE4',   # XRP blue
    }

    nice_names = {
        'BTCUSDT': 'BTC/USDT',
        'ETHUSDT': 'ETH/USDT',
        'DOGEUSDT': 'DOGE/USDT',
        'BNBUSDT': 'BNB/USDT',
        'XRPUSDT': 'XRP/USDT',
    }

    # ─── Figure 1: Price Time Series ───────────────────────────────────
    fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 3 * len(COINS)), sharex=True)
    fig.suptitle('Hourly Close Prices (Jan 2024 – Mar 2026)', fontsize=14, fontweight='bold')

    for i, coin in enumerate(COINS):
        ax = axes[i]
        df = cleaned_data[coin]
        ax.plot(df.index, df['close'], color=colors[coin], linewidth=0.5, alpha=0.9)
        ax.set_ylabel(f'{nice_names[coin]}\nPrice (USDT)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig1_prices.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig1_prices.png")

    # ─── Figure 2: Excess Return Time Series ──────────────────────────
    fig, axes = plt.subplots(len(COINS), 1, figsize=(14, 3 * len(COINS)), sharex=True)
    fig.suptitle('Hourly Excess Returns (Jan 2024 – Mar 2026)', fontsize=14, fontweight='bold')

    for i, coin in enumerate(COINS):
        ax = axes[i]
        ret = returns_dict[coin]
        ax.plot(ret.index, ret['excess_return'], color=colors[coin], linewidth=0.3, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_ylabel(f'{nice_names[coin]}\nExcess Return', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Add stats annotation
        mu = ret['excess_return'].mean()
        sigma = ret['excess_return'].std()
        ax.text(0.02, 0.95, f'μ={mu:.6f}, σ={sigma:.4f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig2_excess_returns.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig2_excess_returns.png")

    # ─── Figure 3: Return Distribution (Histograms) ──────────────────
    fig, axes = plt.subplots(1, len(COINS), figsize=(4 * len(COINS), 4))
    fig.suptitle('Distribution of Hourly Excess Returns', fontsize=14, fontweight='bold')

    for i, coin in enumerate(COINS):
        ax = axes[i]
        ret = returns_dict[coin]['excess_return']
        ax.hist(ret, bins=100, color=colors[coin], alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.set_title(nice_names[coin], fontsize=10)
        ax.set_xlabel('Return', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(labelsize=7)

        # Add kurtosis and skewness
        skew = ret.skew()
        kurt = ret.kurtosis()
        ax.text(0.05, 0.95, f'Skew={skew:.2f}\nKurt={kurt:.1f}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig3_return_distributions.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig3_return_distributions.png")

    # ─── Figure 4: Cumulative Excess Returns ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Cumulative Excess Returns (Jan 2024 – Mar 2026)', fontsize=14, fontweight='bold')

    for coin in COINS:
        ret = returns_dict[coin]['excess_return']
        cum_ret = (1 + ret).cumprod() - 1
        ax.plot(cum_ret.index, cum_ret * 100, label=nice_names[coin],
                color=colors[coin], linewidth=1)

    ax.set_ylabel('Cumulative Excess Return (%)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'fig4_cumulative_returns.png'), dpi=150, bbox_inches='tight')
    print("  ✅ Saved fig4_cumulative_returns.png")

    plt.close('all')
    print()


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Summary Statistics Table
# ═══════════════════════════════════════════════════════════════════════

def print_summary_stats(returns_dict):
    """Print a summary statistics table for the report."""
    print("=" * 60)
    print("SUMMARY STATISTICS (Hourly Excess Returns)")
    print("=" * 60)

    stats = []
    for coin in COINS:
        ret = returns_dict[coin]['excess_return']
        stats.append({
            'Coin': coin.replace('USDT', '/USDT'),
            'Observations': len(ret),
            'Mean (hourly)': f"{ret.mean():.7f}",
            'Std (hourly)': f"{ret.std():.5f}",
            'Skewness': f"{ret.skew():.3f}",
            'Kurtosis': f"{ret.kurtosis():.1f}",
            'Min': f"{ret.min():.4f}",
            'Max': f"{ret.max():.4f}",
            'Ann. Return (%)': f"{ret.mean() * 24 * 365 * 100:.1f}",
            'Ann. Vol (%)': f"{ret.std() * np.sqrt(24 * 365) * 100:.1f}",
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    # Save as CSV for report
    stats_df.to_csv(os.path.join(DATA_DIR, 'summary_statistics.csv'), index=False)
    print("\n  ✅ Saved summary_statistics.csv")
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  COMP0051 - TASK 1: DATA DOWNLOAD AND VISUALIZATION")
    print("═" * 60 + "\n")

    # Step 1: Download
    all_data = download_all_coins()

    # Step 2: Clean
    cleaned_data = clean_all_data(all_data)

    # Step 3: Risk-free rate & excess returns
    print("=" * 60)
    print("STEP 3: Risk-free rate and excess returns")
    print("=" * 60)
    rf = get_risk_free_rate("2024-01-01", "2026-03-25")
    returns_dict = compute_excess_returns(cleaned_data, rf)
    print()

    # Step 4: Explain why r^f can be ignored
    print_rf_explanation(returns_dict)

    # Step 5: Plot
    plot_returns(returns_dict, cleaned_data)

    # Step 6: Summary stats
    print_summary_stats(returns_dict)

    print("=" * 60)
    print("  TASK 1 COMPLETE!")
    print(f"  All files saved in: {DATA_DIR}")
    print("=" * 60)
