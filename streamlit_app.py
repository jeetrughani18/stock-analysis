"""
Stock Performance Analysis — Streamlit Web App
================================================
Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import yfinance as yf

from main import (
    BENCHMARK_MAP,
    download_data,
    align_and_compute_returns,
    compute_all_metrics,
)

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Stock Performance Analysis",
    page_icon="📈",
    layout="centered",
)

# ── Header ───────────────────────────────────
st.title("📈 Stock Performance Analysis")
st.markdown(
    "Analyse any stock against a benchmark index. "
    "Enter the details below and click **Analyse**."
)

# ── Sidebar inputs ───────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    stock_symbol = st.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        placeholder="e.g. RELIANCE.NS",
    ).strip().upper()

    bench_name = st.selectbox(
        "Benchmark Index",
        options=list(BENCHMARK_MAP.keys()),
        index=0,
    )

    risk_free_rate = st.number_input(
        "Risk-Free Rate (annual)",
        min_value=0.0,
        max_value=1.0,
        value=0.07,
        step=0.01,
        format="%.4f",
        help="Enter as a decimal, e.g. 0.07 for 7 %",
    )

    analyse_btn = st.button("🚀 Analyse", use_container_width=True, type="primary")

# ── Main area ────────────────────────────────
if analyse_btn:
    if not stock_symbol:
        st.error("Please enter a stock symbol.")
        st.stop()

    bench_ticker = BENCHMARK_MAP[bench_name]

    # ── Download data ─────────────────────────
    with st.spinner(f"Downloading data for **{stock_symbol}** and **{bench_name}**…"):
        stock_prices = download_data(stock_symbol)
        bench_prices = download_data(bench_ticker)

    if stock_prices is None and bench_prices is None:
        st.error(f"Could not download data for **{stock_symbol}** and **{bench_name}**. Check the symbols.")
        st.stop()
    elif stock_prices is None:
        st.error(f"Could not download data for stock **{stock_symbol}**. Check the symbol.")
        st.stop()
    elif bench_prices is None:
        st.error(f"Could not download data for benchmark **{bench_name}**.")
        st.stop()

    # ── Align & compute ──────────────────────
    with st.spinner("Aligning dates & computing returns…"):
        df = align_and_compute_returns(stock_prices, bench_prices)

    if len(df) < 2:
        st.error(f"Only {len(df)} aligned trading days available — need at least 2.")
        st.stop()

    with st.spinner("Computing metrics…"):
        result = compute_all_metrics(
            df, df["stock_price"], df["bench_price"],
            risk_free_rate, stock_symbol, bench_name,
        )

    # ── Summary cards ─────────────────────────
    st.success(f"Analysis complete for **{stock_symbol}** — {len(df)} trading days")

    col1, col2, col3 = st.columns(3)
    # Pull a few headline numbers from the result DataFrame
    def _val(metric_name: str):
        row = result.loc[result["Metric"] == metric_name, "Value"]
        return row.values[0] if not row.empty else "N/A"

    col1.metric("Rolling 1Y Beta", _val("Rolling_1Y_Beta"))
    col2.metric("Calmar Ratio 1Y", _val("Calmar_Ratio_1Y"))
    col3.metric("Sortino Ratio 1Y", _val("Sortino_Ratio_1Y"))

    # ── Price chart (last 5 years, unadjusted) ──
    st.subheader("📉 Price History — Last 5 Years (Normalised)")
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=5)

    # Download unadjusted Close prices for accurate chart
    def _raw_close(ticker: str) -> pd.Series:
        raw = yf.download(ticker, start=str(start_date.date()),
                          auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw["Close"].dropna()

    raw_stock = _raw_close(stock_symbol)
    raw_bench = _raw_close(bench_ticker)
    chart_prices = pd.DataFrame({
        "stock": raw_stock, "bench": raw_bench,
    }).dropna()

    if len(chart_prices) >= 2:
        norm_stock = chart_prices["stock"] / chart_prices["stock"].iloc[0] * 100
        norm_bench = chart_prices["bench"] / chart_prices["bench"].iloc[0] * 100

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=norm_stock.index, y=norm_stock,
            name=stock_symbol, line=dict(color="#636EFA"),
        ))
        fig.add_trace(go.Scatter(
            x=norm_bench.index, y=norm_bench,
            name=bench_name, line=dict(color="#EF553B"),
        ))
        fig.update_layout(
            yaxis_title="Normalised Value (Base = 100)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for the price chart.")

    # ── Full metrics table ────────────────────
    st.subheader("📊 All Metrics")
    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
        height=min(len(result) * 38 + 40, 900),
    )

    # ── CSV download ──────────────────────────
    csv_data = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_performance_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    # Show a placeholder when no analysis has been run yet
    st.info("👈 Enter a stock symbol in the sidebar and click **Analyse** to get started.")
