"""Backtest engine of the trend-filtered 4-ETF portfolio (2270 Financial Modeling group assignment).

GENERATED FILE: these functions are the same source as the function cells of the notebook
FM_trend_portfolio.ipynb, so the notebook and the Streamlit app run identical code.
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

def download_monthly_prices(tickers, start, end, retries=3, pause=5):
    """Daily adjusted closes from Yahoo Finance (auto_adjust=True: split- and dividend-adjusted)
    -> last available close of each calendar month. `end` is INCLUSIVE here
    (yfinance's own `end` argument is exclusive)."""
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    missing = tickers
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end_excl, interval="1d",
                              auto_adjust=True, progress=False)
            px = raw["Close"]
        except Exception as e:                       # network error / rate limit
            print(f"Download attempt {attempt + 1} failed: {e}")
            px = pd.DataFrame(columns=tickers)
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0])
        missing = [t for t in tickers if t not in px.columns or px[t].isna().all()]
        if not missing:
            break
        if attempt + 1 < retries:
            print(f"No data for {missing}, retrying in {pause}s ...")
            time.sleep(pause)
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no data for {missing}. "
                           "Re-run the cell (Yahoo rate limits are common).")
    px = px[tickers]
    if px.index.tz is not None:
        px = px.tz_localize(None)
    monthly = px.groupby(px.index.to_period("M")).last()          # last close of each month
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "date"
    return monthly


def rf_from_irx(irx_pct, periods_per_year=12):
    """^IRX is the 13-week T-bill yield in % p.a. The yield observed at month-end t is the
    (known) return on cash during month t+1, converted to a monthly rate."""
    y = irx_pct.ffill() / 100
    return ((1 + y) ** (1 / periods_per_year) - 1).shift(1).rename("rf")


def make_fixed_rule(weights):
    """Constant target weights (used for the 60/40 benchmark)."""
    w = pd.Series(weights, dtype=float)
    def rule(prices_hist):
        return w.copy()
    return rule


def make_trend_rule(base_weights, window=10, use_filter=True):
    """Faber (2007)-style trend filter. At each month-end, an asset keeps its base weight if its
    price is above its `window`-month simple moving average; otherwise its weight goes to cash.
    `prices_hist` contains prices up to and including the decision date only."""
    base = pd.Series(base_weights, dtype=float)
    def rule(prices_hist):
        if not use_filter:
            return base.copy()
        p = prices_hist[base.index].iloc[-window:]
        full = (len(p) == window) & p.notna().all()     # require a complete window
        above = p.iloc[-1] > p.mean()                    # month-end price vs moving average
        return base.where(above & full, 0.0)
    return rule


def rebalance_flags(dates, rule):
    """True on the dates where the portfolio is rebalanced (always on the first date)."""
    flags = np.zeros(len(dates), dtype=bool)
    if rule != "BH":
        freq = {"M": "M", "Q": "Q", "A": "Y"}[rule]
        per = np.asarray(dates.to_period(freq))
        flags[:-1] = per[:-1] != per[1:]               # last observation of each period
    flags[0] = True                                    # initial investment
    return pd.Series(flags, index=dates)


def run_backtest(prices, weight_fn, rebalance="M", tc_bps=0.0, rf=None, start=None):
    """Event-driven monthly backtest.
    prices : month-end prices (may include a warm-up period before `start`)
    start  : first month whose return is recorded. The first decision is taken at the last
             month-end before `start`, using data up to that date only.
    rf     : per-period return earned by the cash weight, indexed by return date
    Returns (res, weights): res has gross/net returns, turnover and cost per return date;
    weights are the asset weights HELD during each return period (decided one period earlier)."""
    rets  = prices.pct_change(fill_method=None)
    dates = prices.index
    i0 = 0
    if start is not None:
        i0 = int(dates.searchsorted(pd.Timestamp(start))) - 1
        if i0 < 0:
            raise ValueError("No month-end before `start`: cannot take the first decision.")
    trade_dates = dates[i0:]
    flags = rebalance_flags(trade_dates, rebalance)
    tc    = tc_bps / 10_000
    w     = pd.Series(0.0, index=prices.columns)       # start in cash
    rows, held, warn = [], [], []

    for k in range(len(trade_dates) - 1):
        t, t_next = trade_dates[k], trade_dates[k + 1]
        turnover = 0.0
        if flags.iloc[k]:
            target = weight_fn(prices.loc[:t]).reindex(prices.columns).fillna(0.0)
            turnover = float((target - w).abs().sum())
            w = target

        r = rets.loc[t_next]
        missing = (w != 0) & r.isna()
        if missing.any():                              # held asset without a return: flag it
            warn.append((t_next.strftime("%Y-%m"), list(r.index[missing])))
        r = r.fillna(0.0)

        rf_t = 0.0 if rf is None else rf.get(t_next, np.nan)
        if pd.isna(rf_t):
            warn.append((t_next.strftime("%Y-%m"), "risk-free missing"))
            rf_t = 0.0
        cash_w = 1.0 - float(w.sum())
        gross  = float((w * r).sum() + cash_w * rf_t)
        cost   = turnover * tc                          # paid at the trade, at month-end t
        net    = (1 + gross) * (1 - cost) - 1

        rows.append((t_next, gross, net, turnover, cost))
        held.append(w.rename(t_next))
        w = w * (1 + r) / (1 + gross)                  # weights drift until the next rebalance

    if warn:
        print(f"WARNING: {len(warn)} data issues (returns set to 0):", warn[:5])
    res = pd.DataFrame(rows, columns=["date", "gross", "net", "turnover", "cost"]).set_index("date")
    return res, pd.DataFrame(held)


def run_all(prices, rf, base_weights, bench_weights, ma_window=10, use_filter=True,
            tc_bps=10.0, bench_tc_bps=0.0, rebalance="M", start=None, ref_weights=None):
    """Runs the strategy, the unfiltered static mix, the benchmark and (optionally) a reference
    portfolio on the same sample. Benchmark and reference are shown gross of costs.
    This is the single entry point used by both the notebook and the Streamlit app."""
    if use_filter and start is not None:
        n_hist = int(prices.index.searchsorted(pd.Timestamp(start)))   # month-ends up to the first decision
        if n_hist < ma_window:
            raise ValueError(f"A {ma_window}-month moving average needs {ma_window} month-ends before {start}, "
                             f"but the data have {n_hist}. Start the download earlier or shorten the window.")
    strat, weights = run_backtest(prices, make_trend_rule(base_weights, ma_window, use_filter),
                                  rebalance, tc_bps, rf, start)
    static, _ = run_backtest(prices, make_trend_rule(base_weights, ma_window, False),
                             rebalance, tc_bps, rf, start)
    bench, _ = run_backtest(prices, make_fixed_rule(bench_weights),
                            rebalance, bench_tc_bps, rf, start)
    out = {"strategy": strat, "weights": weights, "static": static,
           "benchmark": bench, "rf": rf.reindex(strat.index)}
    if ref_weights is not None:
        out["reference"], _ = run_backtest(prices, make_fixed_rule(ref_weights),
                                           rebalance, bench_tc_bps, rf, start)
    return out


def perf_stats(r, rf, bench=None, ppy=12):
    """Performance measures of a periodic simple-return series (definitions in the notebook)."""
    ex = r - rf
    wealth = (1 + r).cumprod()
    out = {
        "Avg. return (ann.)":     r.mean() * ppy,
        "CAGR":                   wealth.iloc[-1] ** (ppy / len(r)) - 1,
        "Volatility (ann.)":      r.std() * np.sqrt(ppy),
        "Sharpe ratio (ann.)":    ex.mean() * ppy / (ex.std() * np.sqrt(ppy)),
        "Max drawdown":           (wealth / wealth.cummax() - 1).min(),
        "Worst 12-month return":  (1 + r).rolling(ppy).apply(np.prod, raw=True).min() - 1,
    }
    if bench is not None:
        bex = bench - rf
        fit = sm.OLS(ex.values, sm.add_constant(bex.values)).fit()
        alpha, beta = fit.params
        sigma_eps = np.sqrt(fit.mse_resid)
        active = r - bench
        out.update({
            "Beta":                     beta,
            "Alpha (ann.)":             alpha * ppy,
            "Alpha t-stat":             fit.tvalues[0],
            "R-squared":                fit.rsquared,
            "Info. ratio (regression)": alpha / sigma_eps * np.sqrt(ppy),
            "Tracking error (ann.)":    active.std() * np.sqrt(ppy),
            "Info. ratio (TE)":         active.mean() * ppy / (active.std() * np.sqrt(ppy)),
        })
    return pd.Series(out)


PCT_ROWS = ["Avg. return (ann.)", "CAGR", "Volatility (ann.)", "Max drawdown",
            "Worst 12-month return", "Alpha (ann.)", "Tracking error (ann.)",
            "Annual turnover", "Avg. cash weight", "Cost drag (ann.)"]

def format_table(t, pct_rows=PCT_ROWS):
    """Percentages for return-type rows, 2 decimals for ratios."""
    f = t.copy().astype(object)
    for row in t.index:
        f.loc[row] = [("" if pd.isna(v) else (f"{v:.2%}" if row in pct_rows else f"{v:.2f}"))
                      for v in t.loc[row]]
    return f


def cum_index(r, start_date):
    """Cumulative value of 1 invested at start_date (the series starts at exactly 1)."""
    return pd.concat([pd.Series([1.0], index=[start_date]), (1 + r).cumprod()])


def calendar_returns(r):
    """Compounded return per calendar year."""
    return (1 + r).groupby(r.index.year).prod() - 1

