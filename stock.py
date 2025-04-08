import yfinance as yf
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def download_data(ticker):
    filename = f"{ticker}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)  # 👈 ADD parse_dates=True
    else:
        df = yf.Ticker(ticker).history(period="max")
        df.to_csv(filename)
    df.index = pd.to_datetime(df.index, utc=True)  # 👈 Add utc=True to avoid mixed tz warning
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors='ignore')
    return df

def add_features(df):
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df = df.loc["1990-01-01":].copy()
    return df

def add_horizon_features(df, horizons):
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_averages["Close"]
        df[f"Trend_{horizon}"] = df.shift(1).rolling(horizon).sum()["Target"]
    df = df.dropna(subset=df.columns[df.columns != "Tomorrow"])
    return df

def add_pattern_features(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(20).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    return df.dropna()

def predict(train, test, predictors, model, thresholds):
    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= thresholds).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(df, model, pattern_model, predictors, pattern_features, start=2500, step=250):
    all_predictions = []
    for i in range(start, df.shape[0], step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i + step)].copy()

        pattern_probs = pattern_model.predict_proba(test[pattern_features])[:, 1]
        thresholds = [0.55 if p < 0.5 else 0.45 for p in pattern_probs]

        predictions = predict(train, test, predictors, model, thresholds)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def run_for_stock(ticker):
    print(f"Processing {ticker}...")
    df = download_data(ticker)
    df = add_features(df)
    df = add_pattern_features(df)

    horizons = [2, 5, 60, 250, 1000]
    df = add_horizon_features(df, horizons)

    predictors = [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]
    pattern_features = ["SMA_20", "SMA_50", "Volatility", "Momentum"]

    model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=1)
    pattern_model = joblib.load(f"{ticker}_pattern_model.pkl")

    df = df.dropna()
    predictions = backtest(df, model, pattern_model, predictors, pattern_features)
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"{ticker} Precision: {precision:.4f}")
    return predictions

# Run for S&P 500, Apple, Microsoft, IBM
tickers = ["^GSPC", "AAPL", "MSFT", "IBM"]
results = {}

for ticker in tickers:
    results[ticker] = run_for_stock(ticker)
