# KeynesAI - Stock Market Analysis Platform

KeynesAI is a sophisticated web-based platform for stock market analysis, predictions, and portfolio management. Built with Flask and powered by machine learning algorithms, it provides users with tools to analyze stocks, make predictions, and track their investment portfolio.

## Features

- **User Authentication System**
  - Secure login and registration
  - Password hashing for enhanced security
  - Session management

- **Stock Analysis**
  - Real-time stock data analysis
  - Pattern recognition
  - Technical indicators
  - Machine learning-based predictions

- **Portfolio Management**
  - Track multiple stocks
  - Real-time portfolio value calculation
  - Performance metrics
  - Gain/loss tracking

- **Sector Analysis**
  - Hierarchical stock categorization
  - Sector-wise stock browsing
  - Category-based filtering

- **Prediction Engine**
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
- Create a database named 'KeynesAI'
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please open an issue in the GitHub repository. 