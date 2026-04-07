# Enterprise Time Series Forecasting: Statistical, ML & Deep Learning

This repository is a professional-grade collection of time series forecasting implementations, ranging from classical statistical methods to state-of-the-art Deep Learning architectures. It is designed to handle both univariate and multivariate data, with a specific focus on industrial applications like demand forecasting and financial volatility.

## 📁 Repository Structure
The project is categorized into three strategic phases:
1.  **Classical Methods:** Implementation of ETS and ARIMA frameworks.
2.  **Machine Learning & DL:** Utilizing non-linear models (SVM, RF) and sequence-based Neural Networks (CNN, LSTM, GRU).
3.  **Specialized Content:** Advanced modeling using GARCH for finance and Facebook Prophet for automated business forecasting.
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

| Model | Application
| :--- | :---
| **Naive Forecast** | Baseline comparison ($y_{t+1} = y_t$) 
| **SMA / EWMA** | Smoothing noise and identifying trends
| **Holt-Linear** | Forecasting with linear trends 
| **Holt-Winters** | Triple smoothing (Level + Trend + Seasonality)
---

## 📊 Statistical Modeling (ARIMA & VAR)
Advanced linear modeling for complex temporal dependencies.

* **ARIMA (p, d, q):** Leveraging Autoregressive and Moving Average components. Includes manual selection via **ACF/PACF** plots and automated tuning via **Auto-ARIMA**.
* **Vector Models (VAR/VARMA):** Capturing interactions between multiple variables to predict the future.
---

## 🤖 Machine Learning for Time Series
Applying supervised learning for both forecasting and classification.
* **Regression & Trees:** Utilizing Random Forest and Gradient Boosting for non-linear extrapolation.
* **Classification:** Predicting discrete states (e.g., Stock direction or Human activity Recognition (HAR)).
---

## 🧠 Deep Learning Architectures
* **CNN (1D Convolutions):** Efficiently extracting local temporal patterns.
* **RNN (LSTM/GRU):** Solving the vanishing gradient problem for long-term memory dependencies.
* **Human Activity Recognition (HAR):** Classification using multivariate smartphone accelerometer data.
---

## 💎 Advanced & Financial Time Series
* **GARCH:** Essential for financial risk management; models volatility clustering in price data.
* **Facebook Prophet:** A robust Bayesian model for business data with multiple seasonalities and holiday effects.
---

## 🧪 Evaluation & Validation
To prevent data leakage, models are validated using **Walk Forward Validation** (Expanding/Sliding Windows).
---

## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/Varun100000/Enterprise-Time-Series-Forecasting.git](https://github.com/Varun100000/Enterprise-Time-Series-Forecasting.git)

# Install dependencies
pip install statsmodels pmdarima arch prophet scikit-learn tensorflow
```
---

## 1. Notebooks - Foundational Concepts & Baselines
Understanding the nature of time series data, stationarity, and simple benchmarks.
* [01.gaussian random walk with drift.ipynb](./notebooks/01.gaussian%20random%20walk%20with%20drift%20-%20sampling%20log%20returns%20from%20noise%20looks%20like%20stock%20price.ipynb): Simulating stock prices using log returns and noise.
* [02.Naive Forecasting.ipynb](./notebooks/02.Naive%20Forecasting.ipynb): Implementing the $y_{t+1} = y_t$ baseline.
* [03.SimpleMovingAverage.ipynb](./notebooks/03.SimpleMovingAverage.ipynb): Smoothing data to identify underlying trends.
* [04.EWMA.ipynb](./notebooks/04.EWMA.ipynb): Exponentially Weighted Moving Averages for more responsive smoothing.
* [08 - Walk forward validation.ipynb](./notebooks/08%20-%20Walk%20forward%20validation.ipynb): Robust backtesting logic for time-series models.
* [13.Stationarity in Code.ipynb](./notebooks/13.Stationarity%20in%20Code.ipynb): Testing for stationarity using ADF and transformations.
---

## 2. Notebooks - Exponential Smoothing (ETS)
Models that decompose series into Level, Trend, and Seasonality.
* [05.Forecast - SimpleExponentialSmoothing.ipynb](./notebooks/05.Forecast%20-%20SimpleExponentialSmoothing.ipynb): Best for data with no clear trend.
* [06.Forecast - Holts Linear Model.ipynb](./notebooks/06.Forecast%20-%20Holts%20Linear%20Model.ipynb): Handling series with linear trends.
* [07.Forecast - Holt Winters.ipynb](./notebooks/07.Forecast%20-%20Holt%20Winters.ipynb): Triple smoothing for seasonal data.
* [09.Forecast - Holt Winters - Sales Data example.ipynb](./notebooks/09.Forecast%20-%20Holt%20Winters%20-%20Sales%20Data%20example.ipynb): Industrial application on sales datasets.
* [10.Forecast - Holt Winters Stock Prediction.ipynb](./notebooks/10.Forecast%20-%20Holt%20Winters%20Stock%20Prediction.ipynb): Applying ETS to financial price movements.
* [11.Forecast - StateSpaceMode.ipynb](./notebooks/11.Forecast%20-%20StateSpaceMode.ipynb): Advanced state-space representations of time series.
---

## 3. Notebooks - Statistical Modeling (ARIMA & VAR)
Linear models for autocorrelation and multivariate dependencies.
* [12.ARIMA in Code.ipynb](./notebooks/12.ARIMA%20in%20Code.ipynb): Manual ARIMA implementation.
* [14. ACF and PACF in code.ipynb](./notebooks/14.%20ACF%20and%20PACF%20in%20code.ipynb): Identifying (p, d, q) orders visually.
* [15. Auto Arima in Code.ipynb](./notebooks/15.%20Auto%20Arima%20in%20Code.ipynb): Automated hyperparameter tuning using `pmdarima`.
* [16. Auto Arima Stocks.ipynb](./notebooks/16.%20Auto%20Arima%20Stocks.ipynb): Optimizing ARIMA for stock data.
* [17.ACF and PACF - Stock Returns.ipynb](./notebooks/17.ACF%20and%20PACF%20-%20Stock%20Returns%20-%20without%20AutoArima.ipynb): Manual analysis of financial returns.
* [18. Auto ARIMA AND Manual ACF, PACF Sales Data.ipynb](./notebooks/18.%20Auto%20ARIMA%20AND%20Manual%20ACF%2C%20PACF%20Sales%20Data.ipynb): Comparing automated vs. manual modeling on sales data.
* [19.Forecast - VARMA vs VAR vs ARIMA -Temperature.ipynb](./notebooks/19.Forecast%20-%20VARMA%20vs%20VAR%20vs%20ARIMA%20-Temperature.ipynb): Multivariate temperature forecasting.
* [20. Forecast - VARMA vs VAR vs ARIMA - Econometrics.ipynb](./notebooks/20.%20Forecast%20-%20VARMA%20vs%20VAR%20vs%20ARIMA%20-%20Econometrics.ipynb): Applying vector models to economic indicators.
* [21. Granger Causailty.ipynb](./notebooks/21.%20Granger%20Causailty.ipynb): Testing if one series helps predict another.
---

## 4. Notebooks - Machine Learning for Time Series
Applying Supervised Learning models to temporal data.
* [22.Checking Extrapolation Capability.ipynb](./notebooks/22.Checking%20Extrapolation%20Capability%20of%20ML%20Models%20using%20just%20the%20Stock%20Prices%20and%20Not%20Log%20Returns.ipynb): Why raw prices fail in ML without proper features.
* [23.Forecasting - ML Airline Passengers without Differencing.ipynb](./notebooks/23.Forecasting%20-%20ML%20Airline%20Passengers%20without%20Differencing.ipynb): Evaluating ML models on non-stationary data.
* [24.Forecasting - ML Airline Passengers WITH Differencing.ipynb](./notebooks/24.Forecasting%20-%20ML%20Airline%20Passengers%20WITH%20Differencing.ipynb): Improved ML performance through differencing.
* [25.Forecasting - ML Sales Data without Differencing.ipynb](./notebooks/25.Forecasting%20-%20ML%20Sales%20Data%20without%20Differencing.ipynb): Sales prediction using standard ML features.
* [26.Forecast - ML Stocks with Differencing.ipynb](./notebooks/26.Forecast%20-%20ML%20Stocks%20with%20Differencing.ipynb): Predicting stock changes instead of levels.
* [27. ML - Direction of Stock Price Movement.ipynb](./notebooks/27.%20ML%20-%20Direction%20of%20Stock%20Price%20Movement.ipynb): Binary classification for financial trends.
---

## 5. Notebooks - Deep Learning (ANN, CNN, RNN/LSTM)
Sequence modeling using Neural Networks.
* [28. Forecasting - ANN Airline.ipynb](./notebooks/28.%20Forecasting%20-%20ANN%20Airline.ipynb): Basic Multi-Layer Perceptrons for time series.
* [29. Forecasting - Stocks.ipynb](./notebooks/29.%20Forecasting%20-%20Stocks.ipynb): Deep Learning applied to high-variance stock data.
* [30. Human Activity Recognition - ANN.ipynb](./notebooks/30.%20Multivariate%20Time%20Series%20-%20Human%20Activity%20Recognition%20-%20ANN.ipynb): Classifying movement via smartphone data.
* [31.Forecast - CNN Airline Passengers.ipynb](./notebooks/31.Forecast%20-%20CNN%20Airline%20Passengers.ipynb): 1D Convolutions for temporal pattern recognition.
* [32. Human Activity Recognition - CNN.ipynb](./notebooks/32.%20Multivariate%20Time%20Series%20-%20Human%20Activity%20Recognition%20-%20CNN.ipynb): Using CNNs for complex sequence classification.
* [33. Understanding Shapes in RNN.ipynb](./notebooks/33.%20Understanding%20Shapes%20in%20RNN.ipynb): A tutorial on input/output dimensions for sequence models.
* [34.Forecasting - LSTM Airline Passengers.ipynb](./notebooks/34.Forecasting%20-%20LSTM%20Airline%20Passengers.ipynb): Long Short-Term Memory networks for passenger demand.
* [35. Human Activity Recognition - LSTM.ipynb](./notebooks/35.%20Multivariate%20Time%20Series%20-%20Human%20Activity%20Recognition%20-%20LSTM.ipynb): Sequence modeling for high-accuracy HAR.
---

## 6. Notebooks - Advanced Financial & Bayesian Models
Specialized tools for volatility and automated forecasting.
* [36. Garch - Modeling Volatility.ipynb](./notebooks/36.%20Garch%20-%20Modeling%20Volatility%20of%20Stock%20Returns.ipynb): Modeling heteroscedasticity in financial returns.
* [37.Forecast - Prophet - Store Sales.ipynb](./notebooks/37.Forecast%20-%20Prophet%20-%20Store%20Sales.ipynb): Bayesian additive modeling for retail.
* [38.Forecast - Prophet - Airline Passenger.ipynb](./notebooks/38.Forecast%20-%20Prophet%20-%20Airline%20Passenger.ipynb): Automating trend and seasonality discovery.
* [39.Forecast - Prophet - Stocks.ipynb](./notebooks/39.Forecast%20-%20Prophet%20-%20Stocks.ipynb): Applying Prophet to financial time series.
---

## 🎓 Credits & Acknowledgments
The implementations in this repository were developed as part of my learning journey through the following course:
* **Course Name:** [Time Series Analysis, Forecasting, and Machine Learning]
* **Instructor:** [Lazy Programmer]
* **Platform:** [https://www.udemy.com/course/time-series-analysis/]

*Special thanks to the course creators for providing the theoretical framework and datasets used in these notebooks.*