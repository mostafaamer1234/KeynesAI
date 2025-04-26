from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from static.stock import run_for_stock, predict_future, download_data, add_features, add_pattern_features, add_horizon_features
from static.boomCrash import BoomCrashModel

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('index.HTML')

@app.route('/about_us.html')
def about():  
    return render_template('about_us.html')

@app.route('/predictions.html')
def predictions():
    ticker = request.args.get('ticker', 'AAPL') 

    try:
        df = download_data(ticker)
        df = add_features(df)
        df = add_pattern_features(df)
        df = add_horizon_features(df, [2, 5, 60, 250, 1000])
        df = df.dropna()

        predictors = [f"Close_Ratio_{h}" for h in [2, 5, 60, 250, 1000]] + [f"Trend_{h}" for h in [2, 5, 60, 250, 1000]]
        model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=1)
        model.fit(df[predictors], df["Target"])  

        predictions_df = predict_future(df, model, predictors, ticker)

        if predictions_df is not None:
            table_html = predictions_df.head(20).to_html(classes="prediction-table", border=0)
        else:
            table_html = "<p style='color:red;'>No predictions returned.</p>"

    except Exception as e:
        table_html = f"<p style='color:red;'>Error generating predictions for {ticker}: {e}</p>"

    return render_template('predictions.html', predictions_table=table_html, selected_ticker=ticker)

@app.route('/portfolio.html')
def portfolio():
    portfolio_data = [
        {"name": "AAPL", "shares": 10, "buy_price": 150.00, "current_price": 165.32},
        {"name": "IBM", "shares": 5, "buy_price": 130.00, "current_price": 127.45},
        {"name": "MSFT", "shares": 8, "buy_price": 290.00, "current_price": 312.25},
        {"name": "S&P500", "shares": 20, "buy_price": 4000.00, "current_price": 4180.22},
    ]

    total_value = 0
    total_cost = 0
    top_stock = None
    best_gain = float('-inf')

    for stock in portfolio_data:
        stock["value"] = stock["shares"] * stock["current_price"]
        stock["cost"] = stock["shares"] * stock["buy_price"]
        stock["change_percent"] = round(((stock["current_price"] - stock["buy_price"]) / stock["buy_price"]) * 100, 2)
        total_value += stock["value"]
        total_cost += stock["cost"]
        if stock["change_percent"] > best_gain:
            best_gain = stock["change_percent"]
            top_stock = stock["name"]

    gain_loss = round(total_value - total_cost, 2)

    return render_template("portfolio.html",
                           portfolio=portfolio_data,
                           total_value=round(total_value, 2),
                           gain_loss=gain_loss,
                           top_stock=top_stock)

@app.route('/trending_stocks.html')
def trending_stocks():
    return render_template('trending_stocks.html')

@app.route('/nick.html')
def nick():
    return render_template('nick.html')

@app.route('/sam.html')
def sam():
    return render_template('sam.html')

@app.route('/wilson.html')
def wilson():
    return render_template('wilson.html')

@app.route('/mostafa.html')
def mostafa():
    return render_template('mostafa.html')

@app.route('/style.css')
def style():
    return send_from_directory('static', 'style.css')

if __name__ == '__main__':
    app.run(debug=True)
