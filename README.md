# Trend-filtered multi-asset ETF portfolio

Group assignment for **2270 Financial Modeling**: an investor prospectus for a simple, rules-based portfolio,
backtested in Python, with an interactive Streamlit app.

- **Streamlit app:** https://financialmodelinggp-8mn85ltyh3prwzyudx7eqd.streamlit.app/
- **Notebook:** [`FM_trend_portfolio.ipynb`](FM_trend_portfolio.ipynb) ·
  [open in Colab](https://colab.research.google.com/github/cedrotwelve/FinancialModelingGP/blob/main/FM_trend_portfolio.ipynb)

## Strategy

| | |
|---|---|
| Target client | Retail investor aged 50–60, saving for retirement over 5–10 years; wants to beat inflation but would sell in panic after a ~40% fall |
| Assets | SPY (US equities), EFA (developed ex-US equities), IEF (7–10y US Treasuries), GLD (gold); cash in 3-month T-bills (`^IRX`) |
| Rule | 25% in each asset. At every month-end, an asset priced below its 10-month simple moving average moves to cash for the next month (Faber, 2007) |
| Benchmarks | 60/40 SPY/IEF (alpha, beta, information ratio) and the same 25% mix without the filter; S&P 500 (SPY) shown as a reference only |
| Sample | Monthly returns Jan 2006 – Dec 2025; 2005 used only to compute the first moving averages |

## Files

| File | Content |
|---|---|
| `FM_trend_portfolio.ipynb` | Full backtest; reproduces every number, table and figure in the prospectus. Runs top to bottom in a fresh Google Colab session |
| `backtest_core.py` | Data download, allocation rule, backtest engine and performance measures. Same code as the function cells of the notebook |
| `streamlit_app.py` | Interactive app: change the moving-average window, the transaction cost, or switch the filter off |
| `requirements.txt` | Python dependencies (used by Streamlit Community Cloud) |

With its default settings (filter on, 10-month window, 10 bp costs) the app shows the same numbers as the notebook,
because both call `backtest_core.run_all()`.

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Data are downloaded from Yahoo Finance through `yfinance` at run time; no API key and no data files are needed.

## Reference

Faber, M. T. (2007). A Quantitative Approach to Tactical Asset Allocation. *Journal of Wealth Management*, 9(4), 69–79.

_Educational project. Not investment advice._
