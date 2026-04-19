Trading System

🚀 Overview

QuantSys is a trading system prototype designed to simulate how a real quantitative trading application works.
The system processes market data, builds features, generates trading signals, and presents everything through an interactive dashboard.

🌐 Live Application

👉 https://nitinb0109.github.io/Highprep_Zetheta_13041/quantsys_dashboard.html
The dashboard provides a system-level view of:


Data pipeline

Feature engineering
Signal generation
Portfolio workflow


🧠 System Modules

📊 Data Pipeline
Ingests OHLCV market data
Handles missing values and outliers
Adjusts prices for splits and dividends

⚙️ Feature Engine
Builds 40+ financial features:
Technical indicators (RSI, MACD)
Price & momentum features
Volatility metrics
Volume-based signals

📉 Signal Engine
Generates trading signals using:
Factor-based logic
Machine learning predictions

💼 Portfolio Layer
Assigns weights to selected stocks
Supports basic long/short allocation
Rebalancing logic included

📋 Execution (Simulated)
Tracks trade flow in paper trading mode
Logs trades for analysis

📊 Dashboard
Built using HTML/CSS
Displays:
Pipeline stages
Data quality checks
Feature summary
System structure

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost
HTML/CSS
GitHub Pages

📌 Highlights
End-to-end trading workflow simulation
Structured pipeline (data → signals → portfolio)
Clean and simple UI dashboard
Deployed and accessible online
