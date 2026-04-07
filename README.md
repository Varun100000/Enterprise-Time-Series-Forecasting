# Enterprise Time Series Forecasting: Statistical, ML & Deep Learning

This repository is a professional-grade collection of time series forecasting implementations, ranging from classical statistical methods to state-of-the-art Deep Learning architectures. It is designed to handle both univariate and multivariate data, with a specific focus on industrial applications like demand forecasting and financial volatility.

## 📁 Repository Structure
The project is categorized into three strategic phases:
1.  **Classical Methods:** Implementation of ETS and ARIMA frameworks.
2.  **Machine Learning & DL:** Utilizing non-linear models (SVM, RF) and sequence-based Neural Networks (CNN, LSTM, GRU).
3.  **Specialized VIP Content:** Advanced modeling using GARCH for finance and Facebook Prophet for automated business forecasting.

---

## 📖 Table of Contents
1.  [Core Concepts & Data Engineering](#core-concepts--data-engineering)
2.  [Foundational Forecasting (Baseline & ETS)](#foundational-forecasting-baseline--ets)
3.  [Statistical Modeling (ARIMA & VAR)](#statistical-modeling-arima--var)
4.  [Machine Learning for Time Series](#machine-learning-for-time-series)
5.  [Deep Learning Architectures](#deep-learning-architectures)
6.  [Advanced & Financial Time Series](#advanced--financial-time-series)
7.  [Evaluation & Validation](#evaluation--validation)

---

## 🛠 Core Concepts & Data Engineering
Before modeling, the data is pre-processed to ensure mathematical compatibility:
* **Stationarity:** Identifying trends using the **Augmented Dickey-Fuller (ADF) Test**.
* **Transformations:** Log and Box-Cox transforms to handle heteroscedasticity.
* **Feature Engineering:** Implementing $N \times T \times D$ (Samples, Time Steps, Features) formatting for neural networks.

---

## 📈 Foundational Forecasting (Baseline & ETS)
*Exponential Trend Smoothing (ETS)* models error, trend, and seasonal components.

| Model | Application | Notebook Link |
| :--- | :--- | :--- |
| **Naive Forecast** | Baseline comparison ($y_{t+1} = y_t$) | [Naive_Forecasting.ipynb](./Naive_Forecasting.ipynb) |
| **SMA / EWMA** | Smoothing noise and identifying trends | [SimpleMovingAverage.ipynb](./SimpleMovingAverage.ipynb) |
| **Holt-Linear** | Forecasting with linear trends | [Holts_Linear_Model.ipynb](./Holts_Linear_Model.ipynb) |
| **Holt-Winters** | Triple smoothing (Level + Trend + Seasonality) | [Holt_Winters.ipynb](./Holt_Winters.ipynb) |

---

## 📊 Statistical Modeling (ARIMA & VAR)
Advanced linear modeling for complex temporal dependencies.

* **ARIMA (p, d, q):** Leveraging Autoregressive and Moving Average components. Includes manual selection via **ACF/PACF** plots and automated tuning via **Auto-ARIMA**.
* **Vector Models (VAR/VARMA):** Capturing interactions between multiple variables to predict the future.
* **Notebooks:** [ARIMA_Model.ipynb](./ARIMA_Model.ipynb), [VARMA_vs_VAR.ipynb](./VARMA_vs_VAR.ipynb), [Granger_Causality.ipynb](./Granger_Causality.ipynb)

---

## 🤖 Machine Learning for Time Series
Applying supervised learning for both forecasting and classification.
* **Regression & Trees:** Utilizing Random Forest and Gradient Boosting for non-linear extrapolation.
* **Classification:** Predicting discrete states (e.g., Stock direction or User activity).
* **Notebooks:** [ML_Extrapolation.ipynb](./ML_Extrapolation.ipynb), [ML_Airline_Forecasting.ipynb](./ML_Airline_Forecasting.ipynb)

---

## 🧠 Deep Learning Architectures
* **CNN (1D Convolutions):** Efficiently extracting local temporal patterns.
* **RNN (LSTM/GRU):** Solving the vanishing gradient problem for long-term memory dependencies.
* **Human Activity Recognition (HAR):** Classification using multivariate smartphone accelerometer data.
* **Notebooks:** [CNN_LSTM_Models.ipynb](./CNN_LSTM_Models.ipynb), [HAR_Multivariate.ipynb](./HAR_Multivariate.ipynb)

---

## 💎 Advanced & Financial Time Series
* **GARCH:** Essential for financial risk management; models volatility clustering in price data.
* **Facebook Prophet:** A robust Bayesian model for business data with multiple seasonalities and holiday effects.
* **Notebooks:** [Garch_Volatility.ipynb](./Garch_Volatility.ipynb), [Prophet_Forecasting.ipynb](./Prophet_Forecasting.ipynb)

---

## 🧪 Evaluation & Validation
To prevent data leakage, models are validated using **Walk Forward Validation** (Expanding/Sliding Windows).
* **Notebook:** [Walk_Forward_Validation.ipynb](./Walk_Forward_Validation.ipynb)

---

## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/Varun100000/Enterprise-Time-Series-Forecasting.git](https://github.com/Varun100000/Enterprise-Time-Series-Forecasting.git)

# Install dependencies
pip install statsmodels pmdarima arch prophet scikit-learn tensorflow