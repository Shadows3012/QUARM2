import math
import datetime as dt
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import cvxpy as cp
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

start_date = dt.date(2005, 1, 1)
end_date = dt.date.today()

st.set_page_config(
    page_title="Sector Black–Litterman Lab",
    layout="wide",
)

TRADING_DAYS = 252
RNG = np.random.default_rng(42)
COLOR_SEQ = px.colors.qualitative.Plotly

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 1.5rem;
    }
    .big-metric {
        font-size: 1.4rem;
        font-weight: 600;
    }
    .small-caption {
        color: #808080;
        font-size: 0.80rem;
    }
    .stMetric > div {
        padding: 0.4rem 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SECTOR_ETFS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLRE", "XLU", "XLV", "XLY", "XLC",
]

FACTOR_PROXIES = {
    "MKT": "SPY",
    "VAL": "IVE",
    "GRW": "IVW",
    "MOM": "MTUM",
    "QLTY": "QUAL",
    "LVOL": "USMV",
    "SIZE": "IWM",
    "TERM": "TLT",
    "INT": "IEF",
    "CREDIT": "HYG",
    "COM": "DBC",
}

RISK_FREE_PROXY = "SHV"

SECTOR_COLORS = {
    "XLB": "#1f77b4",
    "XLE": "#ff7f0e",
    "XLF": "#2ca02c",
    "XLI": "#d62728",
    "XLK": "#9467bd",
    "XLP": "#8c564b",
    "XLRE": "#e377c2",
    "XLU": "#7f7f7f",
    "XLV": "#bcbd22",
    "XLY": "#17becf",
    "XLC": "#9e9e9e",
}

SECTOR_LABELS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Cons Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Cons Discretionary",
    "XLC": "Comm Services",
}

SECTOR_LABELS_INV = {v: k for k, v in SECTOR_LABELS.items()}


def to_sector_names(tickers: List[str]) -> List[str]:
    return [SECTOR_LABELS.get(t, t) for t in tickers]


@st.cache_data(show_spinner=False)
def load_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data.xs("Adj Close", axis=1, level=0)
        elif "Close" in data.columns.get_level_values(0):
            prices = data.xs("Close", axis=1, level=0)
        else:
            prices = data
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif "Close" in data.columns:
            prices = data["Close"]
        else:
            prices = data
    prices = prices.dropna(how="all")
    available = list(prices.columns)
    keep_cols = [t for t in tickers if t in available]
    missing = [t for t in tickers if t not in available]
    if missing:
        st.warning(f"Missing tickers from Yahoo: {missing}")
    return prices[keep_cols]


def load_aum(tickers: List[str]) -> pd.Series:
    mc = {}
    for t in tickers:
        tk = yf.Ticker(t)
        cap = tk.fast_info.get("market_cap", None)
        if cap is None:
            try:
                cap = tk.info.get("marketCap", None)
            except Exception:
                cap = None
        mc[t] = cap if cap is not None else np.nan
    return pd.Series([mc[t] for t in tickers], index=tickers)


def calculate_mc_weights(prices: pd.DataFrame) -> np.ndarray:
    tickers = list(prices.columns)
    market_caps = load_aum(tickers)
    if market_caps.isna().any():
        market_caps = market_caps.fillna(market_caps.mean())
    weights = market_caps / market_caps.sum()
    return weights.values


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def build_factor_returns(factor_price_df: pd.DataFrame, rf_series: Optional[pd.Series] = None) -> pd.DataFrame:
    rets = compute_returns(factor_price_df)
    factors = {}
    def has(t): return t in rets.columns
    if has(FACTOR_PROXIES["MKT"]):
        mkt = rets[FACTOR_PROXIES["MKT"]]
        if rf_series is not None:
            rf_aligned = rf_series.reindex(rets.index).fillna(0.0)
            factors["MKT"] = mkt - rf_aligned
        else:
            factors["MKT"] = mkt
    if has(FACTOR_PROXIES["VAL"]) and has(FACTOR_PROXIES["GRW"]):
        factors["VAL"] = rets[FACTOR_PROXIES["VAL"]] - rets[FACTOR_PROXIES["GRW"]]
    if has(FACTOR_PROXIES["MOM"]) and has(FACTOR_PROXIES["MKT"]):
        factors["MOM"] = rets[FACTOR_PROXIES["MOM"]] - rets[FACTOR_PROXIES["MKT"]]
    if has(FACTOR_PROXIES["QLTY"]) and has(FACTOR_PROXIES["MKT"]):
        factors["QLTY"] = rets[FACTOR_PROXIES["QLTY"]] - rets[FACTOR_PROXIES["MKT"]]
    if has(FACTOR_PROXIES["LVOL"]) and has(FACTOR_PROXIES["MKT"]):
        factors["LVOL"] = rets[FACTOR_PROXIES["LVOL"]] - rets[FACTOR_PROXIES["MKT"]]
    if has(FACTOR_PROXIES["SIZE"]) and has(FACTOR_PROXIES["MKT"]):
        factors["SIZE"] = rets[FACTOR_PROXIES["SIZE"]] - rets[FACTOR_PROXIES["MKT"]]
    if has(FACTOR_PROXIES["TERM"]) and has(FACTOR_PROXIES["INT"]):
        factors["TERM"] = rets[FACTOR_PROXIES["TERM"]] - rets[FACTOR_PROXIES["INT"]]
    if has(FACTOR_PROXIES["CREDIT"]) and has(FACTOR_PROXIES["INT"]):
        factors["CREDIT"] = rets[FACTOR_PROXIES["CREDIT"]] - rets[FACTOR_PROXIES["INT"]]
    if has(FACTOR_PROXIES["COM"]):
        factors["COM"] = rets[FACTOR_PROXIES["COM"]]
    factor_df = pd.DataFrame(factors).dropna(how="any")
    if factor_df.empty:
        st.error("No valid factor returns could be constructed. Check factor proxies.")
    return factor_df


def estimate_factor_exposures(etf_returns: pd.DataFrame, factor_returns: pd.DataFrame):
    common_idx = etf_returns.index.intersection(factor_returns.index)
    etf_rets = etf_returns.loc[common_idx]
    fact_rets = factor_returns.loc[common_idx]
    betas = pd.DataFrame(index=etf_rets.columns, columns=fact_rets.columns, dtype=float)
    spec_var = pd.Series(index=etf_rets.columns, dtype=float)
    for etf in etf_rets.columns:
        y = etf_rets[etf].dropna()
        X = fact_rets.loc[y.index]
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()
        betas.loc[etf] = model.params[X.columns]
        spec_var[etf] = np.var(model.resid, ddof=len(model.params))
    return betas, spec_var


def build_risk_model(betas: pd.DataFrame, factor_returns: pd.DataFrame, spec_var: pd.Series):
    Sigma_f = factor_returns.cov()
    D = pd.DataFrame(np.diag(spec_var.values), index=spec_var.index, columns=spec_var.index)
    Sigma = betas.values @ Sigma_f.values @ betas.values.T + D.values
    Sigma = pd.DataFrame(Sigma, index=betas.index, columns=betas.index)
    return Sigma, Sigma_f, D


def market_implied_risk_aversion(mkt_returns: pd.Series, rf: Optional[pd.Series] = None) -> float:
    if rf is not None:
        rf_aligned = rf.reindex(mkt_returns.index).fillna(0.0)
        excess = mkt_returns - rf_aligned
    else:
        excess = mkt_returns
    mu = excess.mean()
    var = excess.var()
    if var <= 0:
        return 3.0
    return float(mu / var)


def equilibrium_returns(Sigma: pd.DataFrame, w_mkt: np.ndarray, delta: float) -> np.ndarray:
    return delta * Sigma.values @ w_mkt


def black_litterman_posterior(mu_prior, Sigma, P, Q, Omega, tau):
    if P.shape[0] == 0:
        return mu_prior.copy()
    Sigma_np = Sigma.values
    tauSigma = tau * Sigma_np
    PT = P.T
    middle = np.linalg.inv(P @ tauSigma @ PT + Omega)
    adjustment = tauSigma @ PT @ middle @ (Q - P @ mu_prior)
    return mu_prior + adjustment


def optimize_mean_variance(mu, Sigma, risk_aversion=3.0, long_only=True, te_cap=None, Sigma_te=None, w_b=None):
    n = len(mu)
    Sigma_np = Sigma.values
    w = cp.Variable(n)
    objective = cp.Maximize(w @ mu - 0.5 * risk_aversion * cp.quad_form(w, Sigma_np))
    constraints = [cp.sum(w) == 1.0]
    if long_only:
        constraints.append(w >= 0)
    if te_cap is not None and Sigma_te is not None and w_b is not None:
        dw = w - w_b
        Sigma_te_np = Sigma_te.values
        constraints.append(cp.quad_form(dw, Sigma_te_np) <= te_cap ** 2)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    if w.value is None:
        raise RuntimeError("Optimisation failed.")
    return np.array(w.value).flatten()


def portfolio_stats(mu, w, Sigma):
    Sigma_np = Sigma.values
    ret = float(w @ mu)
    var = float(w @ Sigma_np @ w)
    vol = math.sqrt(max(var, 0.0))
    return {"mu": ret, "vol": vol}


def risk_contributions(w, Sigma):
    Sigma_np = Sigma.values
    var = float(w @ Sigma_np @ w)
    if var <= 0:
        return pd.Series(np.zeros(len(w)), index=Sigma.index)
    sigma_p = math.sqrt(var)
    mcr = Sigma_np @ w / sigma_p
    cr = w * mcr
    return pd.Series(cr, index=Sigma.index)


def tracking_error_and_contrib(w, w_b, Sigma):
    Sigma_np = Sigma.values
    dw = w - w_b
    te_var = float(dw @ Sigma_np @ dw)
    te = math.sqrt(max(te_var, 0.0))
    if te <= 0:
        return 0.0, pd.Series(np.zeros(len(w)), index=Sigma.index)
    mcte = Sigma_np @ dw / te
    cte = dw * mcte
    return te, pd.Series(cte, index=Sigma.index)


def efficient_frontier(mu, Sigma, n_points=25, long_only=True):
    n = len(mu)
    Sigma_np = Sigma.values
    mu_min, mu_max = float(mu.min()), float(mu.max())
    if abs(mu_max - mu_min) < 1e-8:
        w = np.ones(n) / n
        var = float(w @ Sigma_np @ w)
        vol = math.sqrt(max(var, 0.0))
        return pd.DataFrame([{"mu": float(mu_min), "vol": vol, "weights": w}])
    targets = np.linspace(mu_min, mu_max, n_points)
    frontier = []
    for target in targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma_np))
        constraints = [cp.sum(w) == 1.0, w @ mu == target]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS)
            if w.value is not None:
                w_opt = np.array(w.value).flatten()
                var = float(w_opt @ Sigma_np @ w_opt)
                vol = math.sqrt(max(var, 0.0))
                frontier.append({"mu": float(target), "vol": vol, "weights": w_opt})
        except Exception:
            continue
    return pd.DataFrame(frontier)


def factor_variance_contributions(w, w_b, betas, Sigma_f, Sigma):
    B = betas.values
    Sigma_f_np = Sigma_f.values
    Sigma_np = Sigma.values
    factors = list(betas.columns)
    var_total = float(w @ Sigma_np @ w)
    if var_total <= 0:
        zero = pd.Series(0.0, index=factors + ["Specific"])
        return zero, zero
    b_p = w @ B
    factor_var_contr = b_p * (Sigma_f_np @ b_p)
    var_factor = factor_var_contr.sum()
    var_spec = var_total - var_factor
    factor_pos = np.maximum(factor_var_contr, 0.0)
    var_spec = max(var_spec, 0.0)
    denom = factor_pos.sum() + var_spec
    if denom <= 0:
        zero = pd.Series(0.0, index=factors + ["Specific"])
        return zero, zero
    factor_share_total = factor_pos / denom
    specific_share_total = var_spec / denom
    total_series = pd.Series(np.append(factor_share_total, specific_share_total), index=factors + ["Specific"])
    dw = w - w_b
    te_var_total = float(dw @ Sigma_np @ dw)
    if te_var_total <= 0:
        zero = pd.Series(0.0, index=factors + ["Specific"])
        return total_series, zero
    d_b = dw @ B
    factor_te_var_contr = d_b * (Sigma_f_np @ d_b)
    var_factor_te = factor_te_var_contr.sum()
    var_spec_te = te_var_total - var_factor_te
    factor_te_pos = np.maximum(factor_te_var_contr, 0.0)
    var_spec_te = max(var_spec_te, 0.0)
    denom_te = factor_te_pos.sum() + var_spec_te
    if denom_te <= 0:
        zero = pd.Series(0.0, index=factors + ["Specific"])
        return total_series, zero
    factor_share_te = factor_te_pos / denom_te
    specific_share_te = var_spec_te / denom_te
    te_series = pd.Series(np.append(factor_share_te, specific_share_te), index=factors + ["Specific"])
    return total_series, te_series


def compute_drawdown(returns: pd.Series) -> Tuple[pd.Series, float]:
    if returns.empty:
        return pd.Series(dtype=float), 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = float(dd.min())
    return dd, max_dd


def perf_metrics(r: pd.Series, rf: Optional[pd.Series] = None, name: str = "Portfolio") -> Dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {"Name": name, "Ann. Return": np.nan, "Ann. Vol": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}
    if rf is not None:
        rf_aligned = rf.reindex(r.index).fillna(0.0)
        excess = r - rf_aligned
    else:
        excess = r
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std() * math.sqrt(TRADING_DAYS))
    sharpe = float(excess.mean() * TRADING_DAYS) / ann_vol if ann_vol > 0 else np.nan
    _, max_dd = compute_drawdown(r)
    return {"Name": name, "Ann. Return": ann_ret, "Ann. Vol": ann_vol, "Sharpe": sharpe, "Max Drawdown": max_dd}


def rolling_sharpe(r: pd.Series, window: int = 5, rf: Optional[pd.Series] = None) -> pd.Series:
    r = r.dropna()
    if rf is not None:
        rf_aligned = rf.reindex(r.index).fillna(0.0)
        r = r - rf_aligned
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std()
    sharpe = (roll_mean * TRADING_DAYS) / (roll_std * math.sqrt(TRADING_DAYS))
    return sharpe


def stacked_contrib_bar(contrib_pct: pd.Series, title: str) -> go.Figure:
    s = contrib_pct.fillna(0.0)
    total = s.sum()
    if total != 0:
        s = s / total
    labels = s.index.tolist()
    vals = s.values
    fig = go.Figure()
    for i, (name, v) in enumerate(zip(labels, vals)):
        ticker = SECTOR_LABELS_INV.get(name, name)
        color = SECTOR_COLORS.get(ticker, COLOR_SEQ[i % len(COLOR_SEQ)])
        fig.add_trace(go.Bar(x=["Total"], y=[v], name=name, marker_color=color))
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="",
        yaxis_title="Contribution (%)",
        showlegend=True,
        bargap=0.0,
        bargroupgap=0.0,
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def stacked_time_bars(df: pd.DataFrame, index_name: str, title: str) -> go.Figure:
    df = df.fillna(0.0)
    if isinstance(df.index, pd.DatetimeIndex):
        x_vals = df.index
    else:
        x_vals = pd.to_datetime(df.index)
    df_long = df.copy()
    df_long[index_name] = x_vals
    df_long = df_long.melt(id_vars=index_name, var_name="Category", value_name="Value")
    unique_cats = list(df.columns)
    color_map = {}
    for i, c in enumerate(unique_cats):
        ticker = SECTOR_LABELS_INV.get(c, c)
        color_map[c] = SECTOR_COLORS.get(ticker, COLOR_SEQ[i % len(COLOR_SEQ)])
    fig = px.bar(
        df_long,
        x=index_name,
        y="Value",
        color="Category",
        color_discrete_map=color_map,
        category_orders={"Category": unique_cats},
        title=title,
    )
    fig.update_layout(barmode="stack", bargap=0.0, bargroupgap=0.0)
    return fig


def make_pie_chart(series: pd.Series, title: str) -> go.Figure:
    s = series.fillna(0)
    s = s / s.sum() if s.sum() != 0 else s
    labels = list(s.index)
    values = list(s.values)
    colors = []
    factor_idx = 0
    for name in labels:
        if name in SECTOR_LABELS_INV:
            ticker = SECTOR_LABELS_INV[name]
            colors.append(SECTOR_COLORS.get(ticker))
        else:
            colors.append(COLOR_SEQ[factor_idx % len(COLOR_SEQ)])
            factor_idx += 1
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.50,
            sort=False,
            textinfo="label+percent",
            textposition="inside",
            textfont=dict(color="white", size=18),
            marker=dict(colors=colors, line=dict(color="white", width=1)),
            pull=[0] * len(labels),
        )]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", yanchor="top", font=dict(size=22, color="black")),
        showlegend=False,
        height=550,
        margin=dict(l=10, r=10, t=75, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def main():
    st.markdown("<h1 style='margin-bottom:0.2rem;'>Sector Black–Litterman Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p class='small-caption'></p>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Portfolio constraints")
        long_only = st.checkbox("Long-only (no shorts)", value=False)
        use_te_cap = st.checkbox("Impose max tracking error (annualised)?", value=False)
        if use_te_cap:
            te_cap_ann = st.slider("Max tracking error (annualised)", 0.01, 0.20, 0.08, 0.01)
        else:
            te_cap_ann = None
        st.markdown("---")
        st.subheader("General settings")
        tau = st.slider("Tau (τ)", 0.0, 1.0, 0.05, 0.01)
        rf_ann = st.number_input("Risk-free rate (annualised)", 0.0, 0.10, 0.02, 0.005, format="%.3f")
        st.markdown("---")
        st.subheader("Simulation settings (PM history)")
        sim_months = st.slider("Simulation horizon (months)", 12, 120, 60, 12)
        sim_persistence = st.slider("View persistence ρ", 0.0, 0.99, 0.9, 0.01)
        sim_shock_scale = st.slider("Monthly view shock scale", 0.1, 1.0, 0.5, 0.1)

    factor_tickers = list(set(FACTOR_PROXIES.values()))
    all_tickers = sorted(set(SECTOR_ETFS + factor_tickers))
    prices = load_prices(all_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    returns = compute_returns(prices)
    sector_cols = [c for c in prices.columns if c in SECTOR_ETFS]
    if len(sector_cols) == 0:
        st.error("None of the SPDR sector ETFs were found in the downloaded price data.")
        return
    sector_names = to_sector_names(sector_cols)
    prices_sectors = prices[sector_cols]
    etf_returns = returns[sector_cols].dropna(how="all")
    rf_daily = rf_ann / TRADING_DAYS
    rf_returns = pd.Series(rf_daily, index=returns.index)
    factor_cols = [c for c in prices.columns if c in factor_tickers]
    factor_prices = prices[factor_cols]
    factor_returns = build_factor_returns(factor_prices, rf_series=rf_returns)
    if factor_returns.empty:
        st.error("Factor returns empty. Check factor proxies and data window.")
        return
    betas, spec_var = estimate_factor_exposures(etf_returns, factor_returns)
    Sigma, Sigma_f, D = build_risk_model(betas, factor_returns, spec_var)
    Sigma_sector = Sigma.loc[sector_cols, sector_cols]
    betas_sector = betas.loc[sector_cols]
    n = len(sector_cols)
    aum_all = load_aum(sector_cols)
    if aum_all.isna().any():
        aum_all = aum_all.fillna(aum_all.mean())
    X0 = aum_all / aum_all.sum()
    w_bench = X0.values
    mkt_proxy = FACTOR_PROXIES["MKT"]
    if mkt_proxy in returns.columns:
        mkt_ret = returns[mkt_proxy].dropna()
    else:
        st.error("Market proxy (SPY) not found in returns.")
        return
    delta = market_implied_risk_aversion(mkt_ret, rf_returns)
    pi = equilibrium_returns(Sigma_sector, w_bench, delta)
    pi_series = pd.Series(pi, index=sector_cols, name="Equilibrium μ (daily)")

    st.subheader("User views (Black–Litterman)")
    with st.expander("Configure absolute & relative sector views", expanded=False):
        num_abs = st.number_input("Number of absolute views", 0, 15, 0)
        num_rel = st.number_input("Number of relative views", 0, 15, 0)
        confidence_levels = {"High": 0.03, "Medium": 0.06, "Low": 0.10}
        abs_views = []
        rel_views = []
        selected_abs_sectors = set()
        for i in range(num_abs):
            c1, c2, c3 = st.columns(3)
            with c1:
                sector_choice = st.selectbox(f"Abs view {i + 1} – Sector", sector_names, key=f"abs_sector_{i}")
                if sector_choice in selected_abs_sectors:
                    st.warning(f"Sector {sector_choice} is already selected for an absolute view.")
                selected_abs_sectors.add(sector_choice)
                etf = SECTOR_LABELS_INV[sector_choice]
            with c2:
                v_ann = st.number_input(f"Absolute view {i + 1} ", key=f"abs_val_{i}", value=0.06, step=0.01)
            with c3:
                conf = st.selectbox(
                    f"Absolute view {i + 1} – Confidence level",
                    ["High", "Medium", "Low"],
                    index=1,
                    key=f"abs_conf_{i}",
                )
            sd_ann = confidence_levels[conf]
            abs_views.append(
                (etf, v_ann / TRADING_DAYS, sd_ann / math.sqrt(TRADING_DAYS))
            )
        for i in range(num_rel):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                a_name = st.selectbox(f"Relative view {i + 1}: Sector 1", sector_names, key=f"rel_a_{i}")
                a = SECTOR_LABELS_INV[a_name]
            with c2:
                b_name = st.selectbox(f"Relative view {i + 1}: Sector 2", sector_names, key=f"rel_b_{i}")
                b = SECTOR_LABELS_INV[b_name]
                if a_name == b_name:
                    st.warning(f"Sector {a_name} cannot be relative to itself!")
            with c3:
                v_ann = st.number_input(
                    f"Relative view {i + 1}: Sector1 - Sector2 return",
                    key=f"rel_val_{i}",
                    value=0.02,
                    step=0.01,
                )
            with c4:
                conf = st.selectbox(
                    f"Relative view {i + 1} - Confidence level",
                    ["High", "Medium", "Low"],
                    index=1,
                    key=f"rel_conf_{i}",
                )
            sd_ann = confidence_levels[conf]
            rel_views.append(
                (a, b, v_ann / TRADING_DAYS, sd_ann / math.sqrt(TRADING_DAYS))
            )

    K = len(abs_views) + len(rel_views)
    P = np.zeros((K, n))
    Q = np.zeros(K)
    Omega = np.zeros((K, K))
    row = 0
    for etf, val, sd in abs_views:
        idx = sector_cols.index(etf)
        P[row, idx] = 1.0
        Q[row] = val
        Omega[row, row] = max(sd ** 2, 1e-8)
        row += 1
    for a, b, val, sd in rel_views:
        ia = sector_cols.index(a)
        ib = sector_cols.index(b)
        P[row, ia] = 1.0
        P[row, ib] = -1.0
        Q[row] = val
        Omega[row, row] = max(sd ** 2, 1e-8)
        row += 1

    mu_prior = pi.copy()
    mu_bl = black_litterman_posterior(mu_prior, Sigma_sector, P, Q, Omega, tau)
    daily = etf_returns[sector_cols]
    ann_mu_hist = daily.mean() * TRADING_DAYS
    ann_vol_hist = daily.std() * math.sqrt(TRADING_DAYS)
    sector_vol_model_ann = pd.Series(
        np.sqrt(np.diag(Sigma_sector.values)) * math.sqrt(TRADING_DAYS),
        index=sector_names,
        name="Model vol (ann.)",
    )

    frontier_eq = efficient_frontier(pi, Sigma_sector, n_points=40, long_only=long_only)
    frontier_bl = efficient_frontier(mu_bl, Sigma_sector, n_points=40, long_only=long_only)
    if not frontier_eq.empty:
        frontier_eq["mu_ann"] = frontier_eq["mu"] * TRADING_DAYS
        frontier_eq["vol_ann"] = frontier_eq["vol"] * math.sqrt(TRADING_DAYS)
    if not frontier_bl.empty:
        frontier_bl["mu_ann"] = frontier_bl["mu"] * TRADING_DAYS
        frontier_bl["vol_ann"] = frontier_bl["vol"] * math.sqrt(TRADING_DAYS)

    if use_te_cap and te_cap_ann is not None:
        te_cap_daily = te_cap_ann / math.sqrt(TRADING_DAYS)
    else:
        te_cap_daily = None

    w_bl_star = optimize_mean_variance(
        mu_bl,
        Sigma_sector,
        risk_aversion=3.0,
        long_only=long_only,
        te_cap=te_cap_daily,
        Sigma_te=Sigma_sector,
        w_b=w_bench,
    )

    stats_bl_star = portfolio_stats(mu_bl, w_bl_star, Sigma_sector)
    mu_bl_star = stats_bl_star["mu"] * TRADING_DAYS
    vol_bl_star = stats_bl_star["vol"] * math.sqrt(TRADING_DAYS)
    te_bl_star, cte_bl_star = tracking_error_and_contrib(w_bl_star, w_bench, Sigma_sector)

    if not frontier_eq.empty:
        frontier_eq["Sharpe"] = frontier_eq["mu"] / frontier_eq["vol"].replace(0, np.nan)
        idx_eq_star = frontier_eq["Sharpe"].idxmax()
        w_eq_star = np.array(frontier_eq.loc[idx_eq_star, "weights"])
        mu_eq_star = float(frontier_eq.loc[idx_eq_star, "mu_ann"])
        vol_eq_star = float(frontier_eq.loc[idx_eq_star, "vol_ann"])
    else:
        w_eq_star = w_bench.copy()
        mu_eq_star = np.nan
        vol_eq_star = np.nan

    stats_bl_star = portfolio_stats(mu_bl, w_bl_star, Sigma_sector)
    te_bl_star, cte_bl_star = tracking_error_and_contrib(w_bl_star, w_bench, Sigma_sector)
    cr_bl_star = risk_contributions(w_bl_star, Sigma_sector)
    factor_share_total, factor_share_te = factor_variance_contributions(
        w_bl_star, w_bench, betas_sector, Sigma_f, Sigma_sector
    )

    ann_mu_bl_star = stats_bl_star["mu"] * TRADING_DAYS
    ann_vol_bl_star = stats_bl_star["vol"] * math.sqrt(TRADING_DAYS)
    ann_te_bl_star = te_bl_star * math.sqrt(TRADING_DAYS)

    sim_results = None
    if K > 0:
        monthly_idx = etf_returns.resample("M").last().index
        if len(monthly_idx) >= 2:
            max_months = len(monthly_idx)
            n_months = min(sim_months, max_months)
            sim_dates = monthly_idx[-n_months:]
            Q_base = Q.copy()
            Q_curr = Q_base.copy()
            w_path = []
            te_path_ann = []
            te_contrib_path_ann = []
            factor_total_path = []
            factor_te_path = []
            RISK_AVERSION_SIM = 3.0
            te_cap_daily = None
            if te_cap_ann is not None:
                te_cap_daily = te_cap_ann / math.sqrt(TRADING_DAYS)
            for _ in range(n_months):
                eps = RNG.multivariate_normal(
                    mean=np.zeros(K),
                    cov=Omega * (sim_shock_scale ** 2),
                )
                Q_curr = sim_persistence * Q_curr + (1 - sim_persistence) * Q_base + eps
                mu_bl_t = black_litterman_posterior(mu_prior, Sigma_sector, P, Q_curr, Omega, tau)
                w_t = optimize_mean_variance(
                    mu_bl_t,
                    Sigma_sector,
                    risk_aversion=RISK_AVERSION_SIM,
                    long_only=long_only,
                    te_cap=te_cap_daily,
                    Sigma_te=Sigma_sector,
                    w_b=w_bench,
                )
                te_t, cte_t = tracking_error_and_contrib(w_t, w_bench, Sigma_sector)
                cte_abs = np.abs(cte_t.values)
                sum_abs = cte_abs.sum()
                if sum_abs > 0:
                    cte_scaled_daily = cte_abs / sum_abs * te_t
                else:
                    cte_scaled_daily = cte_abs
                fs_total_t, fs_te_t = factor_variance_contributions(
                    w_t, w_bench, betas_sector, Sigma_f, Sigma_sector
                )
                w_path.append(w_t)
                te_path_ann.append(te_t * math.sqrt(TRADING_DAYS))
                te_contrib_path_ann.append(cte_scaled_daily * math.sqrt(TRADING_DAYS))
                factor_total_path.append(fs_total_t.values)
                factor_te_path.append(fs_te_t.values)
            w_path = np.array(w_path)
            te_path_ann = np.array(te_path_ann)
            te_contrib_path_ann = np.array(te_contrib_path_ann)
            factor_total_path = np.array(factor_total_path)
            factor_te_path = np.array(factor_te_path)
            idx_daily = etf_returns.index
            sim_port_ret = pd.Series(index=idx_daily, dtype=float)
            for t in range(len(sim_dates)):
                start = sim_dates[t]
                if t < len(sim_dates) - 1:
                    end = sim_dates[t + 1]
                    mask = (idx_daily >= start) & (idx_daily < end)
                else:
                    mask = (idx_daily >= start)
                if not mask.any():
                    continue
                r_slice = etf_returns.loc[mask, sector_cols] @ w_path[t]
                sim_port_ret.loc[mask] = r_slice.values
            sim_port_ret = sim_port_ret.dropna()
            sim_bench_ret = (etf_returns[sector_cols] @ w_bench).loc[sim_port_ret.index]
            sim_active_ret = sim_port_ret - sim_bench_ret
            sim_perf_port = perf_metrics(sim_port_ret, rf=rf_returns, name="Sim BL portfolio")
            sim_perf_bench = perf_metrics(sim_bench_ret, rf=rf_returns, name="Benchmark (mkt-cap sectors)")
            if not sim_active_ret.empty:
                sim_active_ann_ret = float(sim_active_ret.mean() * TRADING_DAYS)
                sim_active_ann_vol = float(sim_active_ret.std() * math.sqrt(TRADING_DAYS))
                sim_ir_realised = (
                    sim_active_ann_ret / sim_active_ann_vol if sim_active_ann_vol > 0 else np.nan
                )
            else:
                sim_active_ann_ret = np.nan
                sim_active_ann_vol = np.nan
                sim_ir_realised = np.nan
            sim_te_realised = sim_active_ann_vol
            cum_port_sim = (1 + sim_port_ret).cumprod() - 1.0
            cum_bench_sim = (1 + sim_bench_ret).cumprod() - 1.0
            dd_port_sim, _ = compute_drawdown(sim_port_ret)
            dd_bench_sim, _ = compute_drawdown(sim_bench_ret)
            roll_sharpe_port = rolling_sharpe(sim_port_ret, window=5, rf=rf_returns)
            roll_sharpe_bench = rolling_sharpe(sim_bench_ret, window=5, rf=rf_returns)
            sim_results = {
                "dates": sim_dates,
                "w_path": w_path,
                "te_ann": te_path_ann,
                "te_contrib_ann": te_contrib_path_ann,
                "factor_total_path": factor_total_path,
                "factor_te_path": factor_te_path,
                "sim_port_ret": sim_port_ret,
                "sim_bench_ret": sim_bench_ret,
                "sim_active_ret": sim_active_ret,
                "sim_perf_port": sim_perf_port,
                "sim_perf_bench": sim_perf_bench,
                "sim_active_ann_ret": sim_active_ann_ret,
                "sim_active_ann_vol": sim_active_ann_vol,
                "sim_ir_realised": sim_ir_realised,
                "sim_te_realised": sim_te_realised,
                "cum_port_sim": cum_port_sim,
                "cum_bench_sim": cum_bench_sim,
                "dd_port_sim": dd_port_sim,
                "dd_bench_sim": dd_bench_sim,
                "roll_sharpe_port": roll_sharpe_port,
                "roll_sharpe_bench": roll_sharpe_bench,
            }

    tab_snapshot, tab_frontier, tab_sim = st.tabs(
        ["2. BL snapshot", "3. Frontier", "4. Simulation"]
    )

    with tab_snapshot:
        st.markdown("### 🔍 Risk snapshot (current BL Max Sharpe portfolio)")
        vol_ann = ann_vol_bl_star
        te_ann = ann_te_bl_star
        HHI = np.sum(w_bl_star ** 2)
        ENB = 1 / HHI if HHI > 0 else np.nan
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Volatility (ann.)", f"{vol_ann:.2%}" if not np.isnan(vol_ann) else "NA")
        with c2:
            st.metric("Tracking error (ann.)", f"{te_ann:.2%}" if not np.isnan(te_ann) else "NA")
        with c3:
            st.metric("Effective number of bets (ENB)", f"{ENB:.2f}" if not np.isnan(ENB) else "NA")
        st.markdown("---")
        ann_pi_display = pd.Series(pi * TRADING_DAYS, index=sector_names, name="Benchmark μ (ann.)")
        ann_mu_bl_display = pd.Series(mu_bl * TRADING_DAYS, index=sector_names, name="BL μ (ann.)")
        df_mu_comp = pd.DataFrame({
            "Benchmark μ (ann.)": ann_pi_display,
            "BL implied μ (ann.)": ann_mu_bl_display,
        })
        fig_mu_bar = px.bar(
            df_mu_comp,
            barmode="group",
            title="Benchmark vs Black–Litterman implied sector expected returns",
            labels={"value": "Expected return (ann.)", "index": "Sector"},
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        fig_mu_bar.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_mu_bar, use_container_width=True)
        w_bl_series = pd.Series(w_bl_star, index=sector_names, name="BL Max Sharpe")
        w_bench_series = pd.Series(w_bench, index=sector_names, name="Benchmark")
        weights_df = pd.DataFrame({
            "Benchmark": w_bench_series,
            "BL Max Sharpe": w_bl_series,
        })
        fig_w_combined = px.bar(
            weights_df,
            title="Weights (Black–Litterman vs Benchmark)",
            labels={"value": "Weight", "index": "Sector"},
            barmode="group",
            color_discrete_map={"Benchmark": "#636EFA", "BL Max Sharpe": "#EF553B"},
        )
        fig_w_combined.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_w_combined, use_container_width=True)
        delta_weights = w_bl_star - w_bench
        delta_weights_series = pd.Series(delta_weights, index=sector_names, name="Delta Weights")
        fig_delta_weights = px.bar(
            delta_weights_series,
            title="Delta Weights (Black–Litterman vs Benchmark)",
            labels={"value": "Delta Weight", "index": "Sector"},
        )
        fig_delta_weights.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_delta_weights, use_container_width=True)
        if stats_bl_star["vol"] > 0:
            cr_pct = cr_bl_star / stats_bl_star["vol"]
        else:
            cr_pct = cr_bl_star * 0.0
        cr_pct = cr_pct.reindex(sector_cols)
        cr_pct.index = sector_names
        cte_abs = cte_bl_star.abs()
        if cte_abs.sum() > 0:
            cte_pct = cte_abs / cte_abs.sum()
        else:
            cte_pct = cte_abs * 0.0
        cte_pct = cte_pct.reindex(sector_cols)
        cte_pct.index = sector_names
        c1, c2 = st.columns(2)
        with c1:
            fig_cr = make_pie_chart(cr_pct, "Risk contributions by sector")
            st.plotly_chart(fig_cr, use_container_width=True)
        with c2:
            fig_cte = make_pie_chart(cte_pct, "TE contributions by sector")
            st.plotly_chart(fig_cte, use_container_width=True)
        with c1:
            fig_f_total = make_pie_chart(factor_share_total, "Factor contributions to total variance")
            st.plotly_chart(fig_f_total, use_container_width=True)
        with c2:
            fig_f_te = make_pie_chart(factor_share_te, "Factor contributions to TE variance")
            st.plotly_chart(fig_f_te, use_container_width=True)

    with tab_frontier:
        if frontier_eq.empty or frontier_bl.empty:
            st.warning("Efficient frontiers could not be computed for these settings.")
        else:
            fig2d = go.Figure()
            fig2d.add_trace(
                go.Scatter(
                    x=frontier_eq["vol_ann"],
                    y=frontier_eq["mu_ann"],
                    mode="lines",
                    name="Equilibrium frontier (π)",
                    line=dict(color="lightgrey", width=2, dash="dash"),
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=frontier_bl["vol_ann"],
                    y=frontier_bl["mu_ann"],
                    mode="lines",
                    name="BL frontier (μ_BL with views)",
                    line=dict(color="royalblue", width=3),
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=[vol_eq_star],
                    y=[mu_eq_star],
                    mode="markers",
                    name="Equilibrium Max Sharpe",
                    marker=dict(size=11, color="darkgrey", symbol="circle"),
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=[vol_bl_star],
                    y=[mu_bl_star],
                    mode="markers",
                    name="BL Max Sharpe (with views)",
                    marker=dict(size=12, color="red", symbol="circle"),
                )
            )
            ann_pi_front = pd.Series(pi * TRADING_DAYS, index=sector_names)
            ann_mu_bl_front = pd.Series(mu_bl * TRADING_DAYS, index=sector_names)
            fig2d.add_trace(
                go.Scatter(
                    x=sector_vol_model_ann.values,
                    y=ann_pi_front.values,
                    mode="markers+text",
                    name="Sectors (equilibrium μ, model σ)",
                    text=sector_names,
                    textposition="top center",
                    marker=dict(size=8, color="dimgray", symbol="square"),
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=sector_vol_model_ann.values,
                    y=ann_mu_bl_front.values,
                    mode="markers+text",
                    name="Sectors (BL implied μ, model σ)",
                    text=sector_names,
                    textposition="bottom center",
                    marker=dict(size=8, color="royalblue", symbol="triangle-up"),
                )
            )
            fig2d.update_layout(
                title="Benchmark vs BL efficient frontiers (same Σ, different μ)",
                xaxis_title="Volatility (annualised)",
                yaxis_title="Expected return (annualised)",
                legend=dict(orientation="h", y=-0.25),
            )
            fig2d.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig2d, use_container_width=True)

    with tab_sim:
        if K == 0:
            st.info("Define at least one view above to simulate a PM track record.")
        elif sim_results is None:
            st.info("Simulation did not run (check date window / data availability).")
        else:
            dates_sim = sim_results["dates"]
            w_path = sim_results["w_path"]
            te_ann = sim_results["te_ann"]
            te_contrib_path_ann = sim_results["te_contrib_ann"]
            factor_total_path = sim_results["factor_total_path"]
            factor_te_path = sim_results["factor_te_path"]
            sim_active_ann_ret = sim_results["sim_active_ann_ret"]
            sim_te_realised = sim_results["sim_te_realised"]
            sim_ir_realised = sim_results["sim_ir_realised"]
            cum_port_sim = sim_results["cum_port_sim"]
            cum_bench_sim = sim_results["cum_bench_sim"]
            dd_port_sim = sim_results["dd_port_sim"]
            dd_bench_sim = sim_results["dd_bench_sim"]
            roll_sharpe_port = sim_results["roll_sharpe_port"]
            roll_sharpe_bench = sim_results["roll_sharpe_bench"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Active return (ann.)",
                    f"{sim_active_ann_ret:.2%}" if not np.isnan(sim_active_ann_ret) else "NA",
                )
            with c2:
                st.metric(
                    "Realised TE (ann.)",
                    f"{sim_te_realised:.2%}" if sim_te_realised is not None and not np.isnan(sim_te_realised) else "NA",
                )
            with c3:
                st.metric(
                    "Information ratio",
                    f"{sim_ir_realised:.2f}" if not np.isnan(sim_ir_realised) else "NA",
                )

            if len(w_path) > 1:
                turnovers = []
                for t in range(1, len(w_path)):
                    turn = 0.5 * np.sum(np.abs(w_path[t] - w_path[t - 1]))
                    turnovers.append(turn)
                turnovers = np.array(turnovers)
                avg_turn = float(turnovers.mean()) if len(turnovers) > 0 else np.nan
                col_turn, = st.columns(1)
                with col_turn:
                    st.metric(
                        "Average monthly turnover",
                        f"{avg_turn:.2%}" if not np.isnan(avg_turn) else "NA",
                    )
                if len(dates_sim) == len(w_path):
                    turn_index = dates_sim[1:]
                else:
                    turn_index = range(1, len(w_path))
                turn_series = pd.Series(turnovers, index=turn_index, name="Turnover")
                fig_turn = px.line(
                    turn_series,
                    title="Monthly Turnover (simulation)",
                    labels={"value": "Turnover", "index": "Date"},
                    color_discrete_sequence=COLOR_SEQ,
                )
                fig_turn.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_turn, use_container_width=True)
            else:
                st.info("Not enough simulation steps to compute turnover.")

            te_series = pd.Series(te_ann, index=dates_sim, name="TE (ann.)")
            fig_te_line = px.line(
                te_series,
                title="Simulated total tracking error (annualised)",
                labels={"value": "TE (ann.)", "index": "Date"},
                color_discrete_sequence=COLOR_SEQ,
            )
            st.plotly_chart(fig_te_line, use_container_width=True)

            factor_names = list(betas_sector.columns) + ["Specific"]
            f_total_df = pd.DataFrame(
                factor_total_path,
                index=dates_sim,
                columns=factor_names,
            )
            fig_f_total_time = stacked_time_bars(
                f_total_df,
                index_name="Date",
                title="Factor vs specific risk contributions (stacked, sum = 100%)",
            )
            st.plotly_chart(fig_f_total_time, use_container_width=True)
            f_te_df = pd.DataFrame(
                factor_te_path,
                index=dates_sim,
                columns=factor_names,
            )
            fig_f_te_time = stacked_time_bars(
                f_te_df,
                index_name="Date",
                title="Factor vs specific TE contributions (stacked, sum = 100%)",
            )
            st.plotly_chart(fig_f_te_time, use_container_width=True)
            w_df = pd.DataFrame(w_path, index=dates_sim, columns=sector_names)
            w_df = w_df[sector_names]
            fig_w_time = stacked_time_bars(
                w_df,
                index_name="Date",
                title="Simulated BL portfolio weights (stacked, sum = 1)",
            )
            fig_w_time.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_w_time, use_container_width=True)
            st.markdown("#### Cumulative performance vs benchmark (simulated)")
            cum_df_sim = pd.concat(
                [
                    cum_port_sim.rename("Sim BL portfolio"),
                    cum_bench_sim.rename("Benchmark (mkt-cap sectors)"),
                ],
                axis=1,
            ).dropna()
            st.line_chart(cum_df_sim)


if __name__ == "__main__":
    main()
