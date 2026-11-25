"""Streamlit app for the trend-filtered 4-ETF portfolio (2270 Financial Modeling group assignment).

All calculations come from backtest_core.py, which contains the same functions as the notebook
FM_trend_portfolio.ipynb. With the default settings the app reproduces the notebook's numbers.
Run locally with:  streamlit run streamlit_app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest_core as core

# ---- Fixed assumptions: identical to Section 1 of the notebook -------------------
BASE_WEIGHTS  = {"SPY": 0.25, "EFA": 0.25, "IEF": 0.25, "GLD": 0.25}
BENCH_WEIGHTS = {"SPY": 0.60, "IEF": 0.40}   # benchmark: alpha, beta, information ratio
REF_WEIGHTS   = {"SPY": 1.00}                # reference only: US equities alone
RF_TICKER     = "^IRX"
DATA_START, START, END = "2005-01-01", "2006-01-01", "2025-12-31"
REBALANCE     = "M"
BENCH_TC_BPS  = 0
PERIODS_PER_YEAR = 12
CLIENT_PAIN_DD = 0.40
DEFAULT_MA, DEFAULT_TC = 10, 10

STRATEGY_NAME  = "Trend-filtered 4-ETF"
STATIC_NAME    = "Static 25% mix (no filter)"
BENCHMARK_NAME = "60/40 SPY/IEF"
REF_NAME       = "S&P 500 (SPY)"
COLORS = {STRATEGY_NAME: "#1f77b4", STATIC_NAME: "#ff7f0e", BENCHMARK_NAME: "#222222", REF_NAME: "#8c8c8c"}
DASH   = {REF_NAME: "dot"}

st.set_page_config(page_title="Trend-filtered ETF portfolio", page_icon="📈", layout="wide")


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading monthly prices from Yahoo Finance ...")
def load_data():
    raw = core.download_monthly_prices(list(BASE_WEIGHTS) + [RF_TICKER], DATA_START, END)
    return raw[list(BASE_WEIGHTS)], core.rf_from_irx(raw[RF_TICKER], PERIODS_PER_YEAR)


# ---- Sidebar: the assumptions the user can change ------------------------------
st.sidebar.header("Assumptions")
use_filter = st.sidebar.toggle("Trend filter on", value=True,
                               help="Off = hold the static 25% mix, rebalanced monthly.")
ma_window = st.sidebar.slider("Moving-average window (months)", 2, 12, DEFAULT_MA,
                              disabled=not use_filter,
                              help="An asset below its moving average at month-end is held in cash next month. "
                                   "Maximum 12: prices start in 2005.")
tc_bps = st.sidebar.slider("Transaction cost (bp per unit of turnover)", 0, 50, DEFAULT_TC,
                           help="Applied to the strategy and the static mix. The 60/40 benchmark and the S&P 500 are shown gross.")
is_default = use_filter and ma_window == DEFAULT_MA and tc_bps == DEFAULT_TC
st.sidebar.caption("✅ Default settings: same numbers as the prospectus." if is_default
                   else f"Prospectus settings: filter on, {DEFAULT_MA}-month window, {DEFAULT_TC} bp.")

# ---- Header --------------------------------------------------------------------
st.title("Trend-filtered multi-asset ETF portfolio")
st.markdown(
    "25% each in **SPY** (US equities), **EFA** (developed ex-US equities), **IEF** (7–10y US Treasuries) "
    "and **GLD** (gold). At every month-end, an asset priced below its moving average moves to cash "
    "(3-month T-bills) for the next month. Monthly returns, Jan 2006 – Dec 2025. "
    "Benchmark: 60/40 SPY/IEF. The S&P 500 (SPY) is shown for reference: what an all-equity investor experienced.")

try:
    prices, rf_ret = load_data()
except Exception as e:
    st.error(f"Could not download data from Yahoo Finance: {e}. Please reload the page in a minute.")
    st.stop()

out = core.run_all(prices, rf_ret, BASE_WEIGHTS, BENCH_WEIGHTS, ma_window, use_filter,
                   tc_bps, BENCH_TC_BPS, REBALANCE, START, ref_weights=REF_WEIGHTS)
strat, static, bench, rf = (out["strategy"]["net"], out["static"]["net"],
                            out["benchmark"]["net"], out["rf"])
ref = out["reference"]["net"]
series = {STRATEGY_NAME: strat, STATIC_NAME: static, BENCHMARK_NAME: bench, REF_NAME: ref}

stats = pd.DataFrame({
    STRATEGY_NAME:  core.perf_stats(strat,  rf, bench, PERIODS_PER_YEAR),
    STATIC_NAME:    core.perf_stats(static, rf, bench, PERIODS_PER_YEAR),
    BENCHMARK_NAME: core.perf_stats(bench,  rf, None,  PERIODS_PER_YEAR),
    REF_NAME:       core.perf_stats(ref,    rf, None,  PERIODS_PER_YEAR),
})
stats = stats.reindex(core.perf_stats(strat, rf, bench, PERIODS_PER_YEAR).index)

# ---- Headline numbers ----------------------------------------------------------
s, b = stats[STRATEGY_NAME], stats[BENCHMARK_NAME]
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{s['CAGR']:.2%}", f"{s['CAGR'] - b['CAGR']:+.2%} vs 60/40")
c2.metric("Volatility (ann.)", f"{s['Volatility (ann.)']:.2%}",
          f"{s['Volatility (ann.)'] - b['Volatility (ann.)']:+.2%} vs 60/40", delta_color="inverse")
c3.metric("Sharpe ratio", f"{s['Sharpe ratio (ann.)']:.2f}",
          f"{s['Sharpe ratio (ann.)'] - b['Sharpe ratio (ann.)']:+.2f} vs 60/40")
c4.metric("Max drawdown", f"{s['Max drawdown']:.1%}",
          f"{s['Max drawdown'] - b['Max drawdown']:+.1%} vs 60/40")

# ---- Charts --------------------------------------------------------------------
start_date = prices.index[prices.index.get_loc(strat.index[0]) - 1]
cum = pd.DataFrame({k: core.cum_index(v, start_date) for k, v in series.items()})

tab1, tab2, tab3, tab4 = st.tabs(["Cumulative performance", "Drawdowns", "Allocation", "Calendar years"])
with tab1:
    fig = go.Figure()
    for k in cum:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[k], name=k, line=dict(color=COLORS[k], dash=DASH.get(k, "solid"),
                                 width=3 if k == STRATEGY_NAME else 1.8)))
    fig.add_hline(y=1, line=dict(color="grey", width=1))
    fig.update_layout(title=f"Value of 1 invested in {start_date:%b %Y}", yaxis_title="Value",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.15), height=460)
    st.plotly_chart(fig, width="stretch")
with tab2:
    dd = cum / cum.cummax() - 1
    fig = go.Figure()
    for k in dd:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[k], name=k,
                                 line=dict(color=COLORS[k], dash=DASH.get(k, "solid"))))
    fig.add_hline(y=-CLIENT_PAIN_DD, line=dict(color="red", dash="dash"),
                  annotation_text=f"client's pain threshold (-{CLIENT_PAIN_DD:.0%})")
    fig.update_layout(title="Drawdown from the previous peak", yaxis_tickformat=".0%",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.15), height=420)
    st.plotly_chart(fig, width="stretch")
with tab3:
    w = out["weights"]
    fig = go.Figure()
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#d4a017"]
    for a, col in zip(w.columns, palette):
        fig.add_trace(go.Bar(x=w.index, y=w[a], name=a, marker_color=col))
    fig.add_trace(go.Bar(x=w.index, y=1 - w.sum(axis=1), name="Cash (T-bills)", marker_color="#c7c7c7"))
    fig.update_layout(barmode="stack", bargap=0, title="Weights held during each month",
                      yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.15), height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Average cash weight: {(1 - w.sum(axis=1)).mean():.0%} · "
               f"annual turnover: {out['strategy']['turnover'].iloc[1:].mean() * PERIODS_PER_YEAR:.0%} · "
               f"annual cost drag: {out['strategy']['cost'].iloc[1:].mean() * PERIODS_PER_YEAR:.2%}")
with tab4:
    cal = pd.DataFrame({k: core.calendar_returns(v) for k, v in series.items() if k != REF_NAME})
    fig = go.Figure()
    for k in cal:
        fig.add_trace(go.Bar(x=cal.index.astype(str), y=cal[k], name=k, marker_color=COLORS[k]))
    fig.update_layout(barmode="group", title="Calendar-year returns", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.15), height=420)
    st.plotly_chart(fig, width="stretch")

# ---- Performance table -----------------------------------------------------------
st.subheader("Performance measures")
st.dataframe(core.format_table(stats), width="stretch")
st.caption("Monthly simple returns, Jan 2006 – Dec 2025. Strategy and static mix net of the chosen cost; "
           "60/40 benchmark and S&P 500 reference gross. Sharpe, alpha and information ratio use excess returns over the 3-month "
           "T-bill (^IRX). Alpha and beta from an OLS regression on the 60/40 excess return; beta is not "
           "annualized. Means ×12, volatilities ×√12. Blank cells: the 60/40 is the benchmark, so its beta is 1 "
           "and its alpha 0 by definition and the information ratio is not defined; the S&P 500 is shown for "
           "reference only, without regression measures.")

with st.expander("Method, data and limitations"):
    st.markdown(f"""
- **Data:** Yahoo Finance via `yfinance`, daily prices adjusted for splits and dividends, reduced to month-end
  prices. Cash and risk-free: `^IRX` at month-end *t*, earned during month *t+1*.
- **No look-ahead:** the signal uses prices up to month-end *t* only and is applied to the return of month *t+1*.
- **Costs:** {tc_bps} bp per unit of turnover; the initial purchase is included; cash is held at no cost.
- **Limitations:** a backtest is not a forecast. The moving-average window was popularized after studying past
  data; four US-listed ETFs only; returns in USD (a European investor would hold UCITS versions and carry
  EUR/USD risk); taxes and fund-level differences are ignored.
- Source code and notebook: [GitHub repository](https://github.com/cedrotwelve/FinancialModelingGP).
""")
st.caption("Educational project for 2270 Financial Modeling. Not investment advice.")
