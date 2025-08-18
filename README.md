**Key Features**
*RESTful API*: Built with FastAPI, providing a clean and interactive interface for getting predictions.

*Database Integration*: Uses SQLAlchemy and a SQLite database to store prediction history and model performance metrics.

*Historical Tracking*: All predictions are saved to the database, allowing you to track performance and analyze historical data.

*Automated Accuracy Metrics*: The API can fetch the actual closing prices for past predictions and calculate Mean Absolute Percentage Error (MAPE) and other metrics to evaluate model accuracy over time.

*PyTorch LSTM Model*: A LSTM neural network for time-series forecasting.

**API Endpoints**
The following endpoints are available:

`GET /`: A welcome message to confirm the API is running.

`GET /health`: A health check endpoint that returns the status of the API, GPU availability, and whether the model is loaded.

`GET /predict/next-day`: Returns the next-day price prediction for Bitcoin. You can choose whether to save the prediction to the database.

`GET /predictions/history`: Retrieves a list of the most recent predictions from the database.

`GET /predictions/accuracy`: Calculates and returns accuracy metrics (MAE, MAPE) for all predictions where the actual price is known.

`POST /predictions/update-actual-prices`: Fetches the latest Bitcoin prices to update past predictions with the actual closing price.

---

### Prerequisites

You need Python 3.x and Pip installed on your system.

### Installation

1. Clone the repository:
   ```sh
   git clone [https://github.com/Romansolja/BitTorch.git](https://github.com/Romansolja/BitTorch.git)

2. Navigate to the project directory:
     `cd BitTorch`

3. Install the required packages:
     `pip install --user -r requirements.txt`

### Usage
1. Train the Model:

If you don't have a trained model file `best_model.pth`, run the training script first:
  `python main.py`

This will download the latest data, train the model, and save `best_model.pth` in the data/models directory.

2. Run the FastAPI Server:
   ```sh
   uvicorn app.main:app --reload

3. Access the API:

The API will be available at `http://127.0.0.1:8000`. You can access the interactive documentation at `http://127.0.0.1:8000/docs`.

### Example Output

The script will generate three plots showing the price history, training loss, and a comparison of actual vs. predicted prices on the test set.

Console Output:

  `Final Results:`
  `Test MSE (scaled): 0.000159 (baseline: 0.000184)`
  `Test MAE ($): $415.34 (baseline: $444.87)`
  `Test MAPE: 1.25%`

  `Model improvement over baseline: 6.6%`

  `Current BTC price: $34,567.89`
  `Next-day prediction: $34,812.34`
  `Expected change: $244.45 (0.71%)`
  
(Note: Output values are examples and will change each time you run the script.)
