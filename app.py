import os
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from flask import Flask, render_template, request
from io import BytesIO
import base64
from joblib import dump, load
from functools import lru_cache

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    return yf.download(ticker, start=start_date, end=end_date)


def preprocess_data(data):
    """Preprocess and validate stock data"""
    df = data.copy()

    # Feature engineering
    df['Date'] = df.index
    df['Day'] = df['Date'].dt.day.astype('float32')
    df['Month'] = df['Date'].dt.month.astype('float32')
    df['Year'] = df['Date'].dt.year.astype('float32')

    # Create target
    df['Target'] = df['Close'].shift(-1).astype('float32')
    df.dropna(inplace=True)

    # Select features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day', 'Month', 'Year']
    X = df[features].to_numpy().astype('float32')
    y = df['Target'].to_numpy().astype('float32').flatten()

    # Validate shapes
    assert X.ndim == 2 and y.ndim == 1, f"Invalid shapes - X: {X.shape}, y: {y.shape}"

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, df


def train_model(X, y):
    """Train and evaluate Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse


def create_plot(df, prediction=None):
    """Generate stock price visualization"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Plot historical prices
    sns.lineplot(data=df, x='Date', y='Close', label='Historical Price', color='blue')

    # Add prediction marker if available
    if prediction is not None:
        last_date = df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        plt.plot(next_date, prediction, 'ro', markersize=10, label='Predicted Price')

    plt.title('Stock Price History and Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()

    # Save to bytes
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode('utf8')


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    prediction = None
    rmse = None
    last_prices = None
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    if request.method == 'POST':
        ticker = request.form.get('ticker', 'AAPL').upper()
        start_date = request.form.get('start_date', '2020-01-01')
        end_date = request.form.get('end_date', pd.Timestamp.today().strftime('%Y-%m-%d'))

        try:
            # Fetch and process data
            data = fetch_stock_data(ticker, start_date, end_date)
            X, y, scaler, df = preprocess_data(data)

            # Train and predict
            model, rmse = train_model(X, y)
            last_data = X[-1].reshape(1, -1)
            prediction = model.predict(last_data)[0]

            # Generate visualization
            plot_url = create_plot(df, prediction)
            last_prices = df['Close'][-5:].tolist()

            # Cache model
            dump(model, os.path.join(app.config['UPLOAD_FOLDER'], f'{ticker}_model.joblib'))

        except Exception as e:
            error = f"Error: {str(e)}"
            return render_template('index.html', error=error)

    return render_template('index.html',
                           plot_url=plot_url,
                           prediction=round(prediction, 2) if prediction else None,
                           last_prices=last_prices,
                           rmse=round(rmse, 2) if rmse else None,
                           ticker=ticker,
                           start_date=start_date,
                           end_date=end_date)


if __name__ == '__main__':
    app.run(debug=True)
