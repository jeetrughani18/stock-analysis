"""
Stock Performance Analysis Tool
================================
Downloads historical price data via yfinance for a user-specified stock and
benchmark index, then computes a comprehensive set of risk / return metrics
and saves everything to a CSV file.
"""

from __future__ import annotations

import sys
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────
# Benchmark ticker mapping
# ──────────────────────────────────────────────
BENCHMARK_MAP = {
    "NIFTY50": "^NSEI",
    "NIFTY100": "^CNX100",
    "NIFTY200": "^CNX200",
    "NIFTY500": "^CRSLDX",
    "BSE500": "BSE-500.BO",
}

TRADING_DAYS_1Y = 252
TRADING_DAYS_3Y = 252 * 3
TRADING_DAYS_5Y = 252 * 5


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def download_data(ticker: str, period: str = "max") -> pd.Series | None:
    """Download adjusted close prices for *ticker*. Returns None on failure."""
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"[ERROR] Failed to download data for '{ticker}': {e}")
        return None
    if data.empty:
        print(f"[ERROR] No data returned for ticker '{ticker}'. Please check the symbol.")
        return None
    # yfinance may return multi-level columns; flatten if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    series = data["Close"].dropna().rename(ticker)
    if series.empty:
        print(f"[ERROR] No valid Close prices for '{ticker}' after removing missing values.")
        return None
    return series


def align_and_compute_returns(stock_prices: pd.Series,
                               bench_prices: pd.Series) -> pd.DataFrame:
    """Align two price series by date, compute daily returns, drop NaNs."""
    df = pd.DataFrame({
        "stock_price": stock_prices,
        "bench_price": bench_prices,
    }).dropna()

    df["stock_ret"] = df["stock_price"].pct_change()
    df["bench_ret"] = df["bench_price"].pct_change()
    df.dropna(inplace=True)
    return df


def slice_by_years(data, years: int):
    """Slice a Series or DataFrame to the last *years* calendar years."""
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    return data.loc[start_date:]


# ──────────────────────────────────────────────
# Metric calculations
# ──────────────────────────────────────────────

# 1. Rolling 1-Year Beta
def rolling_beta(df: pd.DataFrame, window: int = TRADING_DAYS_1Y) -> pd.Series:
    """Rolling OLS beta of stock vs benchmark over *window* trading days."""
    cov = df["stock_ret"].rolling(window).cov(df["bench_ret"])
    var = df["bench_ret"].rolling(window).var()
    beta = cov / var
    return beta.rename("rolling_beta")


def latest_rolling_beta(df: pd.DataFrame, window: int = TRADING_DAYS_1Y) -> float:
    """Most recent rolling beta value. Falls back to all available data."""
    actual_window = min(window, len(df))
    beta_series = rolling_beta(df, actual_window)
    valid = beta_series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[-1]


# 2. Information Ratio (trailing 1 Year)
def information_ratio(df: pd.DataFrame,
                      window: int = TRADING_DAYS_1Y) -> float:
    """
    IR = (R_p - R_b) / tracking_error
    Computed over the last *window* trading days.
    """
    recent = df.iloc[-window:]
    excess = recent["stock_ret"] - recent["bench_ret"]
    tracking_error = excess.std()
    if tracking_error == 0:
        return np.nan
    # Annualise the mean excess and the tracking error
    return (excess.mean() * TRADING_DAYS_1Y) / (tracking_error * np.sqrt(TRADING_DAYS_1Y))


# 3 & 5. Drawdown helpers
def drawdown_series(prices: pd.Series) -> pd.Series:
    """Return the drawdown series (always ≤ 0) for a price series."""
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    return dd


def max_drawdown(prices: pd.Series, window: int | None = None) -> float:
    """Maximum drawdown over the last *window* trading days (or full series)."""
    if window is not None:
        prices = prices.iloc[-window:]
    return drawdown_series(prices).min()


def average_drawdown(prices: pd.Series, window: int | None = None) -> float:
    """Average drawdown over the last *window* trading days (or full series)."""
    if window is not None:
        prices = prices.iloc[-window:]
    dd = drawdown_series(prices)
    return dd[dd < 0].mean() if (dd < 0).any() else 0.0


# 4. Days to Recovery
def days_to_recovery(prices: pd.Series) -> int:
    """
    Number of calendar trading days from the trough of the maximum drawdown
    back to the previous peak level. Returns -1 if not yet recovered.
    """
    dd = drawdown_series(prices)
    trough_idx = dd.idxmin()
    peak_before_trough = prices.loc[:trough_idx].idxmax()
    peak_value = prices.loc[peak_before_trough]

    after_trough = prices.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_value]
    if recovered.empty:
        return -1  # not yet recovered
    recovery_date = recovered.index[0]
    # Count the trading days between trough and recovery
    return len(prices.loc[trough_idx:recovery_date]) - 1


# 6. Pain Ratio
def pain_ratio(prices: pd.Series, risk_free_rate: float = 0.0,
               window: int | None = None) -> float:
    """
    Pain Ratio = Average Excess Return / Pain Index

    Average Excess Return = Annualised Avg Return − Risk-Free Rate
    Pain Index            = Σ|drawdowns| / number of periods
                          (average of absolute drawdown values over ALL periods)
    """
    if window is not None:
        prices = prices.iloc[-window:]
    avg_ret = prices.pct_change().dropna().mean() * TRADING_DAYS_1Y
    avg_excess_ret = avg_ret - risk_free_rate
    dd = drawdown_series(prices)
    pain_index = dd.abs().mean()          # Σ|dd| / N across all periods
    if pain_index == 0:
        return np.nan
    return avg_excess_ret / pain_index


# 7. Return Spread vs Benchmark (Outperformance Days)
def return_spread(df: pd.DataFrame, window: int | None = None) -> dict:
    """
    Compute daily spread, count outperformance days.
    Returns dict with total_days, outperf_days, outperf_pct.
    """
    recent = df if window is None else df.iloc[-window:]
    spread = recent["stock_ret"] - recent["bench_ret"]
    total = len(spread)
    outperf = int((spread > 0).sum())
    return {
        "total_days": total,
        "outperf_days": outperf,
        "outperf_pct": round(outperf / total * 100, 2) if total else 0.0,
    }


# 8. Calmar Ratio (1 Year)
def calmar_ratio(prices: pd.Series, window: int = TRADING_DAYS_1Y) -> float:
    """Calmar = Annualised Return / |Max Drawdown| over trailing 1 year."""
    recent = prices.iloc[-window:]
    ann_ret = (recent.iloc[-1] / recent.iloc[0]) - 1  # simple 1-year return
    mdd = max_drawdown(recent)
    if mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


# 9. Sortino Ratio
def sortino_ratio(df: pd.DataFrame, risk_free_rate: float,
                  window: int = TRADING_DAYS_1Y) -> float:
    """
    Sortino = (R_p - R_f) / downside_deviation
    R_f is entered as an annual rate and converted to daily.
    """
    recent = df["stock_ret"].iloc[-window:]
    rf_daily = risk_free_rate / TRADING_DAYS_1Y
    excess = recent - rf_daily
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_1Y)
    if downside_dev == 0:
        return np.nan
    ann_excess = excess.mean() * TRADING_DAYS_1Y
    return ann_excess / downside_dev


# ──────────────────────────────────────────────
# Aggregation & CSV export
# ──────────────────────────────────────────────
def compute_all_metrics(df: pd.DataFrame,
                        stock_prices: pd.Series,
                        bench_prices: pd.Series,
                        risk_free_rate: float,
                        stock_symbol: str,
                        bench_name: str) -> pd.DataFrame:
    """Calculate every metric and return a two-column (Metric, Value) DataFrame."""

    metrics: dict[str, object] = {}
    metrics["Stock"] = stock_symbol
    metrics["Benchmark"] = bench_name
    metrics["Risk_Free_Rate"] = risk_free_rate
    metrics["Data_Start"] = str(df.index.min().date())
    metrics["Data_End"] = str(df.index.max().date())
    metrics["Total_Trading_Days"] = len(df)

    # Pre-slice data by actual calendar dates
    stock_1y = slice_by_years(stock_prices, 1)
    stock_3y = slice_by_years(stock_prices, 3)
    stock_5y = slice_by_years(stock_prices, 5)
    bench_1y = slice_by_years(bench_prices, 1)
    bench_3y = slice_by_years(bench_prices, 3)
    bench_5y = slice_by_years(bench_prices, 5)
    df_1y = slice_by_years(df, 1)
    df_3y = slice_by_years(df, 3)
    df_5y = slice_by_years(df, 5)

    # Check which periods have enough data (need ≥ 2 points)
    total_days = len(df)
    has_1y = len(df_1y) >= 2
    has_3y = len(df_3y) > len(df_1y)   # more data than 1Y
    has_5y = len(df_5y) > len(df_3y)   # more data than 3Y

    # If less than 1 year of data, use all available data for 1Y metrics
    if not has_1y:
        stock_1y = stock_prices
        bench_1y = bench_prices
        df_1y = df
        has_1y = len(df_1y) >= 2

    # 1. Rolling Beta (latest — uses all available data)
    beta_val = latest_rolling_beta(df)
    metrics["Rolling_1Y_Beta"] = round(beta_val, 4) if not np.isnan(beta_val) else "N/A"

    # 2. Information Ratio (1Y or available)
    if has_1y:
        metrics["Information_Ratio_1Y"] = round(
            information_ratio(df_1y, window=len(df_1y)), 4)
    else:
        metrics["Information_Ratio_1Y"] = "N/A"

    # 3. Maximum Drawdown
    metrics["Max_Drawdown_1Y_Stock"] = round(max_drawdown(stock_1y), 4) if has_1y else "N/A"
    metrics["Max_Drawdown_3Y_Stock"] = round(max_drawdown(stock_3y), 4) if has_3y else "N/A"
    metrics["Max_Drawdown_5Y_Stock"] = round(max_drawdown(stock_5y), 4) if has_5y else "N/A"

    # 4. Days to Recovery (stock & benchmark, 1Y / 3Y / 5Y)
    metrics["Days_to_Recovery_1Y_Stock"] = days_to_recovery(stock_1y) if has_1y else "N/A"
    metrics["Days_to_Recovery_3Y_Stock"] = days_to_recovery(stock_3y) if has_3y else "N/A"
    metrics["Days_to_Recovery_5Y_Stock"] = days_to_recovery(stock_5y) if has_5y else "N/A"
    metrics["Days_to_Recovery_1Y_Bench"] = days_to_recovery(bench_1y) if has_1y else "N/A"
    metrics["Days_to_Recovery_3Y_Bench"] = days_to_recovery(bench_3y) if has_3y else "N/A"
    metrics["Days_to_Recovery_5Y_Bench"] = days_to_recovery(bench_5y) if has_5y else "N/A"

    # 5. Average Drawdown
    metrics["Avg_Drawdown_1Y_Stock"] = round(average_drawdown(stock_1y), 4) if has_1y else "N/A"
    metrics["Avg_Drawdown_3Y_Stock"] = round(average_drawdown(stock_3y), 4) if has_3y else "N/A"
    metrics["Avg_Drawdown_5Y_Stock"] = round(average_drawdown(stock_5y), 4) if has_5y else "N/A"

    # 6. Pain Ratio
    metrics["Pain_Ratio_1Y"] = round(pain_ratio(stock_1y, risk_free_rate), 4) if has_1y else "N/A"
    metrics["Pain_Ratio_3Y"] = round(pain_ratio(stock_3y, risk_free_rate), 4) if has_3y else "N/A"
    metrics["Pain_Ratio_5Y"] = round(pain_ratio(stock_5y, risk_free_rate), 4) if has_5y else "N/A"

    # 7. Return Spread / Outperformance Days
    for label, df_slice, has_data in [("1Y", df_1y, has_1y),
                                       ("3Y", df_3y, has_3y),
                                       ("5Y", df_5y, has_5y)]:
        if has_data:
            sp = return_spread(df_slice)
            metrics[f"Spread_Total_Days_{label}"] = sp["total_days"]
            metrics[f"Spread_Outperf_Days_{label}"] = sp["outperf_days"]
            metrics[f"Spread_Outperf_Pct_{label}"] = sp["outperf_pct"]
        else:
            metrics[f"Spread_Total_Days_{label}"] = "N/A"
            metrics[f"Spread_Outperf_Days_{label}"] = "N/A"
            metrics[f"Spread_Outperf_Pct_{label}"] = "N/A"

    # 8. Calmar Ratio (1Y or available)
    if has_1y:
        metrics["Calmar_Ratio_1Y"] = round(
            calmar_ratio(stock_1y, window=len(stock_1y)), 4)
    else:
        metrics["Calmar_Ratio_1Y"] = "N/A"

    # 9. Sortino Ratio (1Y or available)
    if has_1y:
        metrics["Sortino_Ratio_1Y"] = round(
            sortino_ratio(df_1y, risk_free_rate, window=len(df_1y)), 4)
    else:
        metrics["Sortino_Ratio_1Y"] = "N/A"

    return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])


def save_to_csv(result: pd.DataFrame, stock_symbol: str) -> str:
    """Save the metrics DataFrame (Metric | Value) to a CSV file."""
    filename = f"{stock_symbol}_performance_metrics.csv"
    result.to_csv(filename, index=False)
    return filename


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def get_user_inputs() -> tuple[str, str, float]:
    """Prompt the user for stock symbol, benchmark, and risk-free rate."""
    stock_symbol = input("Enter Stock Symbol (e.g. RELIANCE.NS): ").strip().upper()

    print("\nAvailable Benchmarks:")
    for key in BENCHMARK_MAP:
        print(f"  - {key}")
    bench_choice = input("Select Benchmark Index: ").strip().upper()
    if bench_choice not in BENCHMARK_MAP:
        sys.exit(f"[ERROR] Invalid benchmark '{bench_choice}'. "
                 f"Choose from {list(BENCHMARK_MAP.keys())}")

    rf_input = input("Enter Risk-Free Rate (annual, e.g. 0.07 for 7%): ").strip()
    try:
        risk_free_rate = float(rf_input)
    except ValueError:
        sys.exit("[ERROR] Risk-free rate must be a number.")

    return stock_symbol, bench_choice, risk_free_rate


def main() -> None:
    stock_symbol, bench_name, risk_free_rate = get_user_inputs()
    bench_ticker = BENCHMARK_MAP[bench_name]

    print(f"\n⏳ Downloading data for {stock_symbol} and {bench_name} ({bench_ticker})…")
    stock_prices = download_data(stock_symbol)
    bench_prices = download_data(bench_ticker)

    if stock_prices is None and bench_prices is None:
        sys.exit(f"[ERROR] Could not download data for both '{stock_symbol}' and '{bench_name}'. Exiting.")
    elif stock_prices is None:
        sys.exit(f"[ERROR] Could not download data for stock '{stock_symbol}'. Exiting.")
    elif bench_prices is None:
        sys.exit(f"[ERROR] Could not download data for benchmark '{bench_name}'. Exiting.")

    print("📐 Aligning dates and computing returns…")
    df = align_and_compute_returns(stock_prices, bench_prices)

    # Also create aligned price series (needed for drawdown functions)
    aligned_stock = df["stock_price"]
    aligned_bench = df["bench_price"]

    if len(df) < 2:
        sys.exit(f"[ERROR] Only {len(df)} aligned trading days available; "
                 f"need at least 2 days to compute returns.")

    print(f"📊 Computing metrics ({len(df)} trading days available)…\n")
    result = compute_all_metrics(
        df, aligned_stock, aligned_bench,
        risk_free_rate, stock_symbol, bench_name,
    )

    # ── Pretty-print to console ──────────────
    print("=" * 60)
    print(f"  Stock Performance Analysis — {stock_symbol}")
    print("=" * 60)
    for _, row in result.iterrows():
        print(f"  {row['Metric']:<35} {row['Value']}")
    print("=" * 60)

    # ── Save CSV ─────────────────────────────
    csv_file = save_to_csv(result, stock_symbol)
    print(f"\n✅ Metrics saved to {csv_file}")


if __name__ == "__main__":
    main()

