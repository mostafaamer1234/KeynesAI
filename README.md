# KeynesAI - Stock Market Analysis Platform

KeynesAI is a Stock predictor that uses Machine learning models for predicting stocks with a 65% accuracy, it is sophisticated web-based platform for stock market analysis, predictions, and portfolio management. Built with Flask and powered by machine learning algorithms, it provides users with tools to analyze stocks, make predictions, and track their investment portfolio.

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
1. Mostafa Amer - Created the Machine learning model, created the sectors page built using tree datastructures for sorting through available stocks, and created the database for user signin/ signup and created the UI Pages for them
2. Nick - Hard codded the boom&crash and chart.py to add as training variables for the machine learning model which increased accuracy up to 65% accuracy.
3. Wilson & sam - Built the UI front-end, and wilson helped Mostafa integrate the machine learning model (back-end) to the front-end.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or support, please open contact us at amermostafa.official477@gmail.com
