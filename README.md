# KeynesAI - Stock Market Analysis Platform

KeynesAI is a Stock predictor that uses a machine learning model to predict next-day stock direction (see the leakage-free evaluation below for real out-of-sample numbers). It is a web-based platform for stock market analysis, predictions, and portfolio management. Built with Flask and powered by machine learning algorithms, it provides users with tools to analyze stocks, make predictions, and track their investment portfolio.

## Features


- **Stock Analysis**
  - Real-time stock data analysis
  - Pattern recognition
  - Technical indicators
  - Machine learning-based predictions
  - Got data for each stock I'm training the model to predict fromn the YFinance library.
  - Then used the data to train the machine learningf model through the machine learning training library RandomTreeClassifier
  
- **User Authentication System**
  - Secure login and registration
  - Password hashing for enhanced security
  - Session management
  - Created a MyPHP databse in XAMPP


- **Portfolio Management**
  - Track multiple stocks
  - Real-time portfolio value calculation
  - Performance metrics
  - Gain/loss tracking

- **Sector Analysis**
  - Hierarchical stock categorization that uses Tree datastructure
  - Sector-wise stock browsing
  - Category-based filtering

- **Prediction Engine**
  - Retrieved data from YFinance to train the model
  - Machine learning-based price predictions
  - Multiple time horizon predictions
  - Technical pattern analysis
  - Random Forest Classifier implementation

## Technical Stack

- **Backend**: Python/Flask
- **Database**: MySQL
- **Machine Learning**: scikit-learn
- **Data Analysis**: pandas
- **Frontend**: HTML, CSS

## Prerequisites

- Python 3.x
- MySQL Server
- XAMPP (for local development)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/KeynesAI.git
cd KeynesAI
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up the database:
- Start XAMPP and ensure MySQL is running
- Create a database named 'KeynesAI' run the follwoing code :
```
CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
```
  
- The application will automatically create the required tables on first run

4. Configure the database connection:
- Update the `db_config` in `app.py` if your MySQL credentials are different

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

## Project Structure

```
KeynesAI/
├── app.py              # Main application file
├── static/            # Static files
│   ├── stock.py       # Stock analysis logic
│   ├── stock_tree.py  # Stock categorization
│   └── boomCrash.py   # Market analysis models
├── templates/         # HTML templates
└── requirements.txt   # Python dependencies
```

## Contributers
1. Mostafa Amer - Created the Machine learning model, Retrieved data from YFinance to train the model, created the sectors page built using tree datastructures for sorting through available stocks, and created the database for user signin/ signup using MYSQL in XAMPP, and created the UI Pages for them
2. Nick - Hard codded the boom&crash and chart.py to add as training variables for the machine learning model which increased accuracy up to 65% accuracy.
3. Wilson & sam - Built the UI front-end, and wilson helped Mostafa integrate the machine learning model (back-end) to the front-end.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please open contact us at amermostafa.official477@gmail.com

This README file was written by: Mostafa Amer


## Honest evaluation (walk-forward, no look-ahead)

The original backtest scored each test window with `*_pattern_model.pkl` models. Those models were trained on the **entire** price history using 5-day-ahead labels, so information from the test period leaked into the predictions, and the reported 65% is not a reliable out-of-sample number.

`evaluation/walk_forward.py` re-evaluates the approach without leakage:
- Every feature at day *t* uses only prices up to *t*. A test enforces this by changing future prices and checking that past features do not move.
- The random forest is retrained yearly on strictly earlier data.
- Results are compared with the naive "always up" baseline and, as a long-or-cash strategy with 5 bp per position change, with buy-and-hold.

| Ticker | Test days (2000–2025) | Accuracy | Always-up baseline | Precision of "up" calls | Strategy CAGR / Sharpe / MaxDD | Buy & hold CAGR / Sharpe / MaxDD |
|---|---|---|---|---|---|---|
| S&P 500 (^GSPC) | 6,353 | 51.8% | 53.6% | 53.8% | -1.2% / 0.01 / -68% | 5.1% / 0.35 / -57% |
| AAPL | 6,354 | 51.4% | 52.2% | 53.3% | 16.9% / 0.73 / -56% | 23.6% / 0.75 / -82% |
| MSFT | 6,354 | 51.5% | 51.0% | 52.2% | 9.4% / 0.50 / -55% | 9.6% / 0.45 / -68% |
| IBM | 6,354 | 49.9% | 51.4% | 51.9% | -0.1% / 0.08 / -64% | 5.4% / 0.33 / -58% |

What this shows:
- **Out of sample, next-day direction is close to a coin flip** for price-only features. Accuracy is below the always-up baseline for 3 of 4 tickers, and the strategy trails buy-and-hold on return.
- **It does cut drawdowns** for AAPL and MSFT, so the most defensible use of the signal is risk management, not return prediction.
- This is the expected result for liquid large caps. The evaluation code is the valuable part: it shows how to test a trading model without fooling yourself.

```bash
pip install pandas numpy scikit-learn pytest
pytest -q evaluation
python evaluation/walk_forward.py
```

*Not investment advice.*
