# ============================================================
# COMMODITY PRICE FORECASTING — STREAMLIT DASHBOARD
# Project by Rayan Rawat
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Price Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("📈 Commodity Price Forecasting Dashboard")
st.markdown("**Energy Futures Analysis | BRENT, WTI, Natural Gas, RBOB, Heating Oil**")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    tickers = {
        'BRENT'      : 'BZ=F',
        'WTI'        : 'CL=F',
        'NATURAL_GAS': 'NG=F',
        'RBOB'       : 'RB=F',
        'HEATING_OIL': 'HO=F'
    }

    data = {}
    for name, symbol in tickers.items():
        try:
            df = yf.download(symbol,
                            start='2010-01-01',
                            progress=False,
                            auto_adjust=True)

            if df.empty:
                st.warning(f"No data returned for {name}")
                continue

            # Fix MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Rename shortened columns
            column_mapping = {
                'C':'Close','H':'High',
                'L':'Low','O':'Open','V':'Volume'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Ensure Close column exists
            if 'Close' not in df.columns:
                df.columns = ['Close','High','Low','Open','Volume']

            df = df.dropna(subset=['Close'])

            # Log returns
            df['Log_Return'] = np.log(
                df['Close'] / df['Close'].shift(1))

            # Regime detection
            rolling_return = df['Log_Return'].rolling(63).mean()
            df['Regime'] = np.where(
                rolling_return >  0.0005,  1,
                np.where(
                rolling_return < -0.0005, -1, 0))
            df['Regime_Label'] = df['Regime'].map(
                {1:'Bull', -1:'Bear', 0:'Choppy'})

            # Rolling volatility
            df['Rolling_Vol'] = (df['Log_Return']
                                .rolling(30).std() * np.sqrt(252) * 100)

            data[name] = df

        except Exception as e:
            st.warning(f"Failed to load {name}: {str(e)}")
            continue

    return data

# Load data with spinner
with st.spinner("Loading commodity data from Yahoo Finance..."):
    data = load_data()

# Safety check
if not data:
    st.error("Failed to load data. Please refresh the page.")
    st.stop()

available_commodities = list(data.keys())

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

selected_commodity = st.sidebar.selectbox(
    "Select Commodity",
    available_commodities,
    index=0
)

date_range = st.sidebar.selectbox(
    "Select Date Range",
    ['Full History (2010-2026)',
     'Last 5 Years',
     'Last 3 Years',
     'Last 1 Year'],
    index=2
)

show_regime = st.sidebar.checkbox(
    "Show Regime Background", value=True)
show_volume = st.sidebar.checkbox(
    "Show Volume", value=False)

# ── Filter by date ────────────────────────────────────────────
df = data[selected_commodity].copy()

if date_range == 'Last 5 Years':
    df = df[df.index >= '2021-01-01']
elif date_range == 'Last 3 Years':
    df = df[df.index >= '2023-01-01']
elif date_range == 'Last 1 Year':
    df = df[df.index >= '2025-01-01']

# ── Key Metrics Row ───────────────────────────────────────────
st.subheader(f"📊 {selected_commodity} — Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

current_price  = df['Close'].iloc[-1]
prev_price     = df['Close'].iloc[-2]
price_change   = current_price - prev_price
pct_change     = (price_change / prev_price) * 100
period_high    = df['Close'].max()
period_low     = df['Close'].min()
current_regime = df['Regime_Label'].iloc[-1]
avg_vol        = df['Rolling_Vol'].iloc[-1]

col1.metric("Current Price",
            f"${current_price:.2f}",
            f"{pct_change:+.2f}%")
col2.metric("Period High",  f"${period_high:.2f}")
col3.metric("Period Low",   f"${period_low:.2f}")
col4.metric("Current Regime", current_regime)
col5.metric("30D Volatility", f"{avg_vol:.1f}%")

st.markdown("---")

# ── Price Chart ───────────────────────────────────────────────
st.subheader("📉 Price History with Regime Classification")

fig = go.Figure()

# Regime background
if show_regime:
    regime_colors = {
         1: 'rgba(76,175,80,0.15)',
        -1: 'rgba(255,87,34,0.15)',
         0: 'rgba(255,249,196,0.15)'
    }
    current_regime_val = df['Regime'].iloc[0]
    start_date = df.index[0]

    for i in range(1, len(df)):
        if df['Regime'].iloc[i] != current_regime_val \
                or i == len(df)-1:
            fig.add_vrect(
                x0=start_date,
                x1=df.index[i],
                fillcolor=regime_colors[current_regime_val],
                layer="below",
                line_width=0
            )
            current_regime_val = df['Regime'].iloc[i]
            start_date = df.index[i]

# Price line
fig.add_trace(go.Scatter(
    x=df.index, y=df['Close'],
    mode='lines',
    name=f'{selected_commodity} Close',
    line=dict(color='#2196F3', width=1.5)
))

fig.update_layout(
    height=450,
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    hovermode='x unified',
    legend=dict(orientation='h',
                yanchor='bottom', y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

# ── Volume Chart ──────────────────────────────────────────────
if show_volume:
    st.subheader("📦 Trading Volume")
    fig_vol_bar = go.Figure()
    fig_vol_bar.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='#9C27B0',
        opacity=0.7
    ))
    fig_vol_bar.update_layout(
        height=300,
        xaxis_title='Date',
        yaxis_title='Volume',
        title=f'{selected_commodity} Trading Volume'
    )
    st.plotly_chart(fig_vol_bar, use_container_width=True)

# ── Returns and Volatility ────────────────────────────────────
st.subheader("📊 Returns & Volatility Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_ret = px.histogram(
        df['Log_Return'].dropna(),
        nbins=100,
        title=f'{selected_commodity} Daily Returns Distribution',
        color_discrete_sequence=['#2196F3']
    )
    fig_ret.update_layout(height=350)
    st.plotly_chart(fig_ret, use_container_width=True)

with col2:
    fig_vol2 = go.Figure()
    fig_vol2.add_trace(go.Scatter(
        x=df.index,
        y=df['Rolling_Vol'],
        mode='lines',
        name='30D Rolling Volatility',
        line=dict(color='#FF5722', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255,87,34,0.1)'
    ))
    fig_vol2.update_layout(
        title=f'{selected_commodity} 30-Day Rolling Volatility (%)',
        height=350,
        yaxis_title='Annualised Volatility (%)'
    )
    st.plotly_chart(fig_vol2, use_container_width=True)

# ── Correlation Heatmap ───────────────────────────────────────
st.subheader("🔗 Cross-Commodity Correlation (Returns)")
st.markdown("---")

returns_df = pd.DataFrame()
for name in data:
    returns_df[name] = data[name]['Log_Return']

corr_matrix = returns_df.corr()

fig_corr = px.imshow(
    corr_matrix,
    color_continuous_scale='RdYlGn',
    zmin=-1, zmax=1,
    title='Return Correlation Matrix',
    text_auto='.2f'
)
fig_corr.update_layout(height=400)
st.plotly_chart(fig_corr, use_container_width=True)

# ── Model Comparison ──────────────────────────────────────────
st.subheader("🤖 Model Performance Comparison")

results = {
    'Naive'  : {'MAE': '-',      'RMSE': '-',      'MAPE': '1.69%'},
    'SARIMA' : {'MAE': '$10.75', 'RMSE': '$13.03', 'MAPE': '14.49%'},
    'Prophet': {'MAE': '$10.70', 'RMSE': '$13.05', 'MAPE': '12.99%'},
    'XGBoost': {'MAE': '$1.03',  'RMSE': '$1.57',  'MAPE': '1.26%'},
    'LSTM'   : {'MAE': '$1.88',  'RMSE': '$2.64',  'MAPE': '2.29%'},
}

results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
st.dataframe(results_df, use_container_width=True)

st.info("""
**Champion Model: XGBoost (MAPE: 1.26%)**
XGBoost outperformed all models including the naive baseline (1.69% MAPE).
SARIMA and Prophet performed worse than naive — consistent with
Efficient Market Hypothesis for daily commodity returns.
""")

# ── Backtest Results ──────────────────────────────────────────
st.subheader("📉 Backtest Results (2023-2026)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Regime + XGBoost Strategy**")
    st.metric("Annual Return",  "0.98%")
    st.metric("Sharpe Ratio",   "0.036")
    st.metric("Max Drawdown",   "-56.76%")
    st.metric("Win Rate",       "20.36%")

with col2:
    st.markdown("**Buy and Hold Benchmark**")
    st.metric("Annual Return",  "6.47%")
    st.metric("Sharpe Ratio",   "0.171")
    st.metric("Max Drawdown",   "-50.30%")
    st.metric("Win Rate",       "52.53%")

st.warning("""
**Finding:** Strategy underperformed buy and hold.
Daily BRENT returns follow near-efficient random walk —
historical price data alone is insufficient for alpha generation.
Alternative data (EIA inventories, options flow) needed for
reliable trading signals.
""")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Project by Rayan Rawat** | NIT Bhopal | 
Ex-Financial Market Analyst at Futures First |
[GitHub](https://github.com/ayan-30/commodity-price-forecasting)
""")