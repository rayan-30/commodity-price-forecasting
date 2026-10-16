# Commodity Price Forecasting
### By Rayan Rawat | NIT Bhopal | ex-Futures First

An end-to-end energy futures price forecasting system 
built on 15 years of daily OHLCV data for BRENT, WTI, 
Natural Gas, RBOB and Heating Oil.

## Live Dashboard
[View Dashboard →](https://commodity-price-forecasting-tc7mivrsfbrw9c89trjszv.streamlit.app/)

## Project Structure
01_data_collection.ipynb → Data pipeline (yfinance, EIA)
02_eda.ipynb → Exploratory analysis
03_modelling.ipynb → 4 forecasting models
04_regime_backtest.ipynb → Regime detection + backtest
05_streamlit_app.py → Interactive dashboard


## Models Built
| Model | MAPE | Notes |
|---|---|---|
| SARIMA | 14.49% | Statistical baseline |
| Prophet | 12.99% | Seasonality-aware |
| XGBoost | 1.26% | Champion model |
| LSTM | 2.29% | Deep learning |
| Naive | 1.69% | Benchmark |

## Key Findings
- XGBoost achieved 1.26% MAPE — 25% better than naive baseline
- BRENT daily returns follow near-efficient random walk
- Natural Gas most volatile commodity — weather driven demand
- BRENT-WTI correlation broke down during 2014 Saudi price war
  and COVID 2020 — correlation is not permanent

## Tech Stack
Python, pandas, numpy, yfinance, statsmodels, Prophet,
XGBoost, TensorFlow/Keras, Plotly, Streamlit

## Domain Context
Energy futures data from NYMEX and ICE exchanges.
Analysis informed by professional experience at 
Futures First — trading BRENT, WTI, Gasoil, RBOB 
and Heating Oil.
