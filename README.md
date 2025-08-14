# BitTorch: Bitcoin Price Prediction with PyTorch

A simple LSTM neural network built with PyTorch to predict the next-day closing price of Bitcoin (BTC-USD).

## Description

This script downloads the last ~2 years of Bitcoin daily prices, trains a small LSTM neural network to predict tomorrow’s closing price from the last 7 days, and compares the model to a simple baseline ("tomorrow ≈ today").

The model reports whether it beats the baseline and visualizes its performance with several plots.

### Key Features:
- **Data Acquisition:** Downloads the latest daily BTC-USD data from Yahoo Finance.
- **Data Processing:** Uses `MinMaxScaler` to normalize prices and correctly splits data into training, validation, and test sets to prevent data leakage.
- **LSTM Model:** A simple, commented LSTM model built with PyTorch.
- **Robust Training:** Implements a full training loop with a baseline comparison, early stopping, and a learning rate scheduler.
- **Evaluation:** Reports key metrics like MSE, MAE (in dollars), and MAPE.
- **Prediction:** Makes a final prediction for the next day's price based on the most recent data.

---

### Prerequisites

You need Python 3.x and Pip installed on your system.

### Installation

1. Clone the repository:
   ```sh
   git clone [https://github.com/Romansolja/BitTorch.git](https://github.com/Romansolja/BitTorch.git)

2. Navigate to the project directory:
     cd BitTorch

3. Install the required packages:
     pip install -r requirements.txt

### Usage

To run the script, execute the main.py file:
  python main.py

The script will automatically download the data, train the model, and display the evaluation plots and the final prediction in the console.

### Example Output

The script will generate three plots showing the price history, training loss, and a comparison of actual vs. predicted prices on the test set.

Console Output:

  `Final Results:
  Test MSE (scaled): 0.000159 (baseline: 0.000184)
  Test MAE ($): $415.34 (baseline: $444.87)
  Test MAPE: 1.25%

  Model improvement over baseline: 6.6%

  Current BTC price: $34,567.89
  Next-day prediction: $34,812.34
  Expected change: $244.45 (0.71%)`
  
(Note: Output values are examples and will change each time you run the script.)
