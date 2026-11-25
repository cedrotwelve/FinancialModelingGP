"""Streamlit app for the trend-filtered 4-ETF portfolio (2270 Financial Modeling group assignment).

All calculations come from backtest_core.py, which contains the same functions as the notebook
FM_trend_portfolio.ipynb. With the prospectus settings the app reproduces the notebook's numbers.
Run locally with:  streamlit run streamlit_app.py
"""
import hashlib
import importlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtest_core as core

# Streamlit Cloud keeps imported modules in memory across code updates: reload the engine so the app
# always runs the backtest_core.py currently in the repository.
core = importlib.reload(core)
with open(core.__file__, "rb") as _f:
    CORE_VERSION = hashlib.md5(_f.read()).hexdigest()       # cached results are dropped if the engine changes

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
PROSPECTUS = {"use_filter": True, "ma_window": 10, "tc_bps": 10}   # settings used in the prospectus

STRATEGY_NAME  = "Trend-filtered 4-ETF"
STATIC_NAME    = "Static 25% mix (no filter)"
BENCHMARK_NAME = "60/40 SPY/IEF"
REF_NAME       = "S&P 500 (SPY)"
PROSPECTUS_NAME = "Strategy, prospectus settings"
COLORS = {STRATEGY_NAME: "#1f4e79", STATIC_NAME: "#e07b39", BENCHMARK_NAME: "#2b2b2b", REF_NAME: "#9aa0a6"}
DASH   = {REF_NAME: "dot"}
ASSET_COLORS = {"SPY": "#1f4e79", "EFA": "#3a8d5c", "IEF": "#7b5ea7", "GLD": "#c9a227"}
ASSET_LABELS = {"SPY": "SPY · US equities", "EFA": "EFA · developed ex-US equities",
                "IEF": "IEF · 7–10y US Treasuries", "GLD": "GLD · gold"}
GITHUB_URL = "https://github.com/cedrotwelve/FinancialModelingGP"
COLAB_URL = ("https://colab.research.google.com/github/cedrotwelve/FinancialModelingGP/blob/main/"
             "FM_trend_portfolio.ipynb")

st.set_page_config(page_title="Trend-filtered ETF portfolio", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: #f2f5f9; border-radius: 10px; padding: 12px 16px;}
.plain-words {background: #f2f5f9; border-left: 4px solid #1f4e79; border-radius: 6px;
              padding: 14px 18px; margin: 6px 0 18px 0; line-height: 1.6;}
.subtitle {color: #566573; font-size: 1.05rem; margin-top: -0.6rem;}
</style>
""", unsafe_allow_html=True)


# ---- Data and calculations (cached) ---------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Downloading monthly prices from Yahoo Finance ...")
def load_data():
    raw = core.download_monthly_prices(list(BASE_WEIGHTS) + [RF_TICKER], DATA_START, END)
    return raw[list(BASE_WEIGHTS)], core.rf_from_irx(raw[RF_TICKER], PERIODS_PER_YEAR)


@st.cache_data(show_spinner=False)
def compute(ma_window, use_filter, tc_bps, core_version):
    prices, rf_ret = load_data()
    return core.run_all(prices, rf_ret, BASE_WEIGHTS, BENCH_WEIGHTS, ma_window, use_filter,
                        tc_bps, BENCH_TC_BPS, REBALANCE, START, ref_weights=REF_WEIGHTS)


def series_of(out):
    return {STRATEGY_NAME: out["strategy"]["net"], STATIC_NAME: out["static"]["net"],
            BENCHMARK_NAME: out["benchmark"]["net"], REF_NAME: out["reference"]["net"]}


def stats_of(out):
    s, rf = series_of(out), out["rf"]
    bench = s[BENCHMARK_NAME]
    t = pd.DataFrame({
        STRATEGY_NAME:  core.perf_stats(s[STRATEGY_NAME], rf, bench, PERIODS_PER_YEAR),
        STATIC_NAME:    core.perf_stats(s[STATIC_NAME],   rf, bench, PERIODS_PER_YEAR),
        BENCHMARK_NAME: core.perf_stats(bench,            rf, None,  PERIODS_PER_YEAR),
        REF_NAME:       core.perf_stats(s[REF_NAME],      rf, None,  PERIODS_PER_YEAR),
    })
    return t.reindex(core.perf_stats(s[STRATEGY_NAME], rf, bench, PERIODS_PER_YEAR).index)


# ---- Sidebar: the assumptions the user can change ------------------------------
for k, v in PROSPECTUS.items():
    st.session_state.setdefault(k, v)

def reset_to_prospectus():
    st.session_state.update(PROSPECTUS)

st.sidebar.header("Assumptions")
st.sidebar.toggle("Trend filter on", key="use_filter",
                  help="Off = hold the static 25% mix, rebalanced monthly.")
st.sidebar.slider("Moving-average window (months)", 2, 12, key="ma_window",
                  disabled=not st.session_state.use_filter,
                  help="An ETF priced below its moving average at month-end is held in T-bills next month. "
                       "Maximum 12: prices start in 2005.")
st.sidebar.slider("Transaction cost (bp per unit of turnover)", 0, 50, key="tc_bps",
                  help="Applied to the strategy and the static mix. The 60/40 benchmark and the S&P 500 are "
                       "shown gross of costs.")
use_filter, ma_window, tc_bps = st.session_state.use_filter, st.session_state.ma_window, st.session_state.tc_bps
is_default = (use_filter, ma_window, tc_bps) == tuple(PROSPECTUS.values())
st.sidebar.button("↺ Reset to prospectus settings", on_click=reset_to_prospectus,
                  disabled=is_default, width="stretch")
st.sidebar.caption("✅ Prospectus settings: the numbers match the prospectus and the notebook." if is_default
                   else "Prospectus settings: filter on, 10-month window, 10 bp. They are shown as a dashed "
                        "line in the charts for comparison.")
st.sidebar.divider()
st.sidebar.markdown("**Who is this for?**  \nA cautious retail investor aged 50–60, saving for retirement over "
                    "5–10 years, who wants to beat inflation but would sell in panic after a 40% fall.")

# ---- Header --------------------------------------------------------------------
st.title("Trend-filtered multi-asset ETF portfolio")
st.markdown('<p class="subtitle">25% each in US equities, developed ex-US equities, US Treasuries and gold, '
            'with a simple trend filter that moves an ETF to T-bills when its price falls below its '
            'moving average. Monthly backtest, January 2006 – December 2025.</p>', unsafe_allow_html=True)

try:
    prices, rf_ret = load_data()
    out = compute(ma_window, use_filter, tc_bps, CORE_VERSION)
    out_p = compute(PROSPECTUS["ma_window"], PROSPECTUS["use_filter"], PROSPECTUS["tc_bps"], CORE_VERSION)
except Exception as e:
    st.error(f"Could not download data from Yahoo Finance: {e}. Please reload the page in a minute.")
    st.stop()

series, stats = series_of(out), stats_of(out)
stats_p = stats_of(out_p)
strat = series[STRATEGY_NAME]
start_date = prices.index[prices.index.get_loc(strat.index[0]) - 1]
cum = pd.DataFrame({k: core.cum_index(v, start_date) for k, v in series.items()})
cum_p = core.cum_index(out_p["strategy"]["net"], start_date)
cal = pd.DataFrame({k: core.calendar_returns(v) for k, v in series.items()})
cash_w = 1 - out["weights"].sum(axis=1)

# ---- In plain words (updates with the settings) ----------------------------------
s, b, r = stats[STRATEGY_NAME], stats[BENCHMARK_NAME], stats[REF_NAME]
setting_txt = (f"trend filter on, {ma_window}-month window" if use_filter else "trend filter off") + f", {tc_bps} bp costs"
mdd = -s["Max drawdown"]
position = ("well within" if mdd < CLIENT_PAIN_DD / 2 else "within" if mdd < CLIENT_PAIN_DD else "beyond")
worst_year = cal[STRATEGY_NAME].idxmin()
cash_txt = (f"On average {cash_w.mean():.0%} of the portfolio was held in T-bills." if use_filter
            else "Without the filter the portfolio is always fully invested.")
st.markdown(f"""
<div class="plain-words"><b>In plain words</b> ({setting_txt}): $1 invested at the end of {start_date:%B %Y}
grew to <b>${cum[STRATEGY_NAME].iloc[-1]:.2f}</b> by {strat.index[-1]:%B %Y}, compared with
${cum[BENCHMARK_NAME].iloc[-1]:.2f} for the 60/40 and ${cum[REF_NAME].iloc[-1]:.2f} for the S&amp;P 500.
The largest fall from a previous peak was <b>{s['Max drawdown']:.0%}</b>, {position} the client's
{CLIENT_PAIN_DD:.0%} pain threshold (60/40: {b['Max drawdown']:.0%}; S&amp;P 500: {r['Max drawdown']:.0%}).
The worst calendar year was {worst_year} ({cal.loc[worst_year, STRATEGY_NAME]:.1%}). {cash_txt}</div>
""", unsafe_allow_html=True)

# ---- Headline numbers ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{s['CAGR']:.2%}", f"{s['CAGR'] - b['CAGR']:+.2%} vs 60/40")
c2.metric("Volatility (ann.)", f"{s['Volatility (ann.)']:.2%}",
          f"{s['Volatility (ann.)'] - b['Volatility (ann.)']:+.2%} vs 60/40", delta_color="inverse")
c3.metric("Sharpe ratio", f"{s['Sharpe ratio (ann.)']:.2f}",
          f"{s['Sharpe ratio (ann.)'] - b['Sharpe ratio (ann.)']:+.2f} vs 60/40")
c4.metric("Max drawdown", f"{s['Max drawdown']:.1%}",
          f"{s['Max drawdown'] - b['Max drawdown']:+.1%} vs 60/40")
if not is_default:
    p = stats_p[STRATEGY_NAME]
    st.caption(f"With the prospectus settings: CAGR {p['CAGR']:.2%} · volatility {p['Volatility (ann.)']:.2%} · "
               f"Sharpe {p['Sharpe ratio (ann.)']:.2f} · max drawdown {p['Max drawdown']:.1%}")


# ---- Charts --------------------------------------------------------------------
def base_layout(fig, title, height=440, pct=False, **kw):
    fig.update_layout(title=title, height=height, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", y=-0.14, x=0), margin=dict(l=10, r=10, t=50, b=10), **kw)
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


def add_prospectus_line(fig, y):
    if not is_default:
        fig.add_trace(go.Scatter(x=y.index, y=y, name=PROSPECTUS_NAME,
                                 line=dict(color="#7fa7cf", width=1.6, dash="dash")))


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "Drawdowns", "How it works", "Allocation", "Calendar years"])

with tab1:
    fig = go.Figure()
    for k in cum:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[k], name=k,
                                 line=dict(color=COLORS[k], dash=DASH.get(k, "solid"),
                                           width=3 if k == STRATEGY_NAME else 1.8)))
    add_prospectus_line(fig, cum_p)
    fig.add_hline(y=1, line=dict(color="#b0b7bf", width=1))
    st.plotly_chart(base_layout(fig, f"Value of $1 invested at the end of {start_date:%b %Y}"), width="stretch")

with tab2:
    dd = cum / cum.cummax() - 1
    fig = go.Figure()
    for k in dd:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[k], name=k,
                                 line=dict(color=COLORS[k], dash=DASH.get(k, "solid"),
                                           width=2.6 if k == STRATEGY_NAME else 1.5)))
    add_prospectus_line(fig, cum_p / cum_p.cummax() - 1)
    fig.add_hline(y=-CLIENT_PAIN_DD, line=dict(color="#c0392b", dash="dash"),
                  annotation_text=f"client's pain threshold (-{CLIENT_PAIN_DD:.0%})",
                  annotation_position="bottom left")
    st.plotly_chart(base_layout(fig, "Fall from the previous peak (drawdown)", pct=True), width="stretch")

with tab3:
    st.markdown("**The rule in three steps**  \n"
                f"1. At every month-end, compare each ETF's price with its average price over the last "
                f"**{ma_window} months**.  \n"
                "2. Above the average → keep its 25%. Below the average → move that 25% to T-bills.  \n"
                "3. Apply the decision to the next month only, then repeat. Only past prices are used.")
    asset = st.selectbox("Choose an ETF", list(BASE_WEIGHTS), format_func=lambda a: ASSET_LABELS[a])
    px_a = prices[asset]
    ma_a = px_a.rolling(ma_window).mean()
    view = slice(start_date, None)
    fig = go.Figure()
    if use_filter:
        held = out["weights"][asset] > 0
        left = held.index - pd.offsets.MonthEnd(1)                 # each holding month runs (left, date]
        run_start = None
        for i, h in enumerate(held.values):
            if not h and run_start is None:
                run_start = left[i]
            if h and run_start is not None:
                fig.add_vrect(x0=run_start, x1=left[i], fillcolor="#c0392b", opacity=0.12, line_width=0)
                run_start = None
        if run_start is not None:
            fig.add_vrect(x0=run_start, x1=held.index[-1], fillcolor="#c0392b", opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=px_a.loc[view].index, y=px_a.loc[view], name=f"{asset} month-end price",
                             line=dict(color=ASSET_COLORS[asset], width=2.4)))
    fig.add_trace(go.Scatter(x=ma_a.loc[view].index, y=ma_a.loc[view], name=f"{ma_window}-month moving average",
                             line=dict(color="#566573", width=1.5, dash="dash")))
    if use_filter:                                                  # legend entry for the shaded months
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Month held in T-bills",
                                 marker=dict(symbol="square", size=12, color="rgba(192,57,43,0.25)")))
    fig.update_xaxes(type="date")
    st.plotly_chart(base_layout(fig, f"{ASSET_LABELS[asset]}: price, moving average and months in T-bills",
                                yaxis_type="log", yaxis_title="Adjusted price (log scale)"), width="stretch")
    if use_filter:
        switches = int(held.astype(int).diff().abs().iloc[1:].sum())
        st.caption(f"{asset} was held in {held.mean():.0%} of the months, with {switches} switches in or out "
                   f"({switches / (len(held) / PERIODS_PER_YEAR):.1f} per year). Each switch costs {tc_bps} bp on "
                   "the amount traded.")
    else:
        st.info("The trend filter is off: every ETF is always held at 25%. Turn it on in the sidebar to see the signals.")

with tab4:
    w = out["weights"]
    fig = go.Figure()
    for a in w.columns:
        fig.add_trace(go.Bar(x=w.index, y=w[a], name=a, marker_color=ASSET_COLORS[a]))
    fig.add_trace(go.Bar(x=w.index, y=cash_w, name="T-bills (cash)", marker_color="#d5d8dc"))
    st.plotly_chart(base_layout(fig, "Weights held during each month", pct=True, barmode="stack", bargap=0),
                    width="stretch")
    st.caption(f"Average cash weight: {cash_w.mean():.0%} · annual turnover: "
               f"{out['strategy']['turnover'].iloc[1:].mean() * PERIODS_PER_YEAR:.0%} · annual cost drag: "
               f"{out['strategy']['cost'].iloc[1:].mean() * PERIODS_PER_YEAR:.2%}")

with tab5:
    fig = go.Figure()
    for k in [STRATEGY_NAME, STATIC_NAME, BENCHMARK_NAME]:
        fig.add_trace(go.Bar(x=cal.index.astype(str), y=cal[k], name=k, marker_color=COLORS[k]))
    st.plotly_chart(base_layout(fig, "Calendar-year returns", pct=True, barmode="group"), width="stretch")

# ---- Performance table -----------------------------------------------------------
st.subheader("Performance measures")
st.dataframe(core.format_table(stats), width="stretch", height=35 * (len(stats) + 1) + 3)   # all rows visible
st.caption("Monthly simple returns, Jan 2006 – Dec 2025. Strategy and static mix net of the chosen cost; "
           "60/40 benchmark and S&P 500 reference gross. Sharpe, alpha and information ratio use excess "
           "returns over the 3-month T-bill (^IRX). Alpha and beta from an OLS regression on the 60/40 excess "
           "return; beta is not annualized. Means ×12, volatilities ×√12. Blank cells: the 60/40 is the "
           "benchmark, so its beta is 1 and its alpha 0 by definition and the information ratio is not "
           "defined; the S&P 500 is shown for reference only, without regression measures.")

# ---- Downloads -------------------------------------------------------------------
monthly = pd.DataFrame({
    "strategy_net": out["strategy"]["net"], "strategy_gross": out["strategy"]["gross"],
    "static_mix_net": out["static"]["net"], "benchmark_60_40": out["benchmark"]["net"],
    "sp500_spy": out["reference"]["net"], "tbill_rf": out["rf"],
})
monthly = monthly.join(out["weights"].add_prefix("weight_")).assign(weight_cash=cash_w)
tag = f"{'ma' + str(ma_window) if use_filter else 'nofilter'}_{tc_bps}bp"
d1, d2 = st.columns(2)
d1.download_button("⬇ Monthly returns and weights (CSV)", monthly.to_csv().encode(),
                   file_name=f"monthly_returns_{tag}.csv", mime="text/csv", width="stretch")
d2.download_button("⬇ Performance measures (CSV)", stats.to_csv().encode(),
                   file_name=f"performance_{tag}.csv", mime="text/csv", width="stretch")

with st.expander("Method, data and limitations"):
    st.markdown(f"""
- **Data:** Yahoo Finance via `yfinance`, daily prices adjusted for splits and dividends, reduced to month-end
  prices. Cash and risk-free: `^IRX` at month-end *t*, earned during month *t+1*.
- **No look-ahead:** the signal uses prices up to month-end *t* only and is applied to the return of month *t+1*.
- **Costs:** {tc_bps} bp per unit of turnover; the initial purchase is included; cash is held at no cost.
- **Limitations:** a backtest is not a forecast. The moving-average window was popularized after studying past
  data; four US-listed ETFs only; returns in USD (a European investor would hold UCITS versions and carry
  EUR/USD risk); taxes and fund-level differences are ignored.
""")
st.caption(f"Educational project for 2270 Financial Modeling · [Source code]({GITHUB_URL}) · "
           f"[Notebook in Colab]({COLAB_URL}) · Not investment advice.")
