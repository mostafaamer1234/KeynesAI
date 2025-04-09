import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from boomCrash import BoomCrashModel  # Import the class from its module


def download_data(ticker):
    filename = f"{ticker}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="max")
        df.to_csv(filename)
    df.index = pd.to_datetime(df.index, utc=True)
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


def backtest(df, model, predictors, ticker, start=2500, step=250):
    all_predictions = []

    try:
        pattern_model = joblib.load(f"{ticker}_pattern_model.pkl")
        # No need to import BoomCrashModel here since we imported it at the top
        boom_crash_model = joblib.load("boom_crash_model.pkl")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None

    pattern_features = ["SMA_20", "SMA_50", "Volatility", "Momentum"]

    for i in range(start, df.shape[0], step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i + step)].copy()

        # Get predictions from both models
        pattern_probs = pattern_model.predict_proba(test[pattern_features])[:, 1]
        thresholds = []

        for date, p in zip(test.index, pattern_probs):
            phase = boom_crash_model.get_market_phase(date)

            # Base threshold
            threshold = 0.6

            # Adjust based on pattern confidence
            if p < 0.4:
                threshold += 0.1
            elif p > 0.6:
                threshold -= 0.1

            # Adjust based on market phase
            if phase == "boom":
                threshold -= 0.05
            elif phase == "crash":
                threshold += 0.05

            thresholds.append(max(0.35, min(0.65, threshold)))  # Keep within bounds

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

    model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=1)
    df = df.dropna()

    predictions = backtest(df, model, predictors, ticker)
    if predictions is not None:
        precision = precision_score(predictions["Target"], predictions["Predictions"])
        print(f"{ticker} Precision: {precision:.4f}")
        return predictions
    return None


if __name__ == "__main__":
    # Check if model files exist in the current directory
    required_files = ["boom_crash_model.pkl"]
    tickers = ["^GSPC", "AAPL", "MSFT", "IBM"]
    required_files.extend([f"{ticker}_pattern_model.pkl" for ticker in tickers])

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Missing model files. Please run:")
        if "boom_crash_model.pkl" in missing_files:
            print("- boomCrash.py to create the boom/crash model")
        if any("_pattern_model.pkl" in f for f in missing_files):
            print("- chart.py to create the pattern models")
    else:
        results = {}
        for ticker in tickers:
            results[ticker] = run_for_stock(ticker)