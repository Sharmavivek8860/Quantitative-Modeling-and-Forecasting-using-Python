# --------------------------------------------------------------------------
# PROJECT 2: Quantitative Modeling and Forecasting using Python
# --------------------------------------------------------------------------
#
# 🔹 Concept:
# This project applies statistical and mathematical modeling to predict trends
# or future outcomes. This Python script is an alternative to the MATLAB
# version, using popular data science libraries to achieve the same goal.
#
# 🔹 Keywords:
# "Regression modeling, forecasting, Python, Scikit-learn, quantitative
# analysis, data modeling, time series."
#
# 🔹 How to Run:
# 1. Open a terminal or command prompt.
# 2. Make sure you are in the same directory as this file.
# 3. Run the command: python quantitative_modeling.py
#
# --------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def quantitative_modeling():
    """
    Performs time-series modeling, forecasting, and evaluation.
    """
    print("Running quantitative modeling and forecasting analysis...")

    # ----------------------------------------------------------------------
    # 1. Generate Synthetic Data
    # In a real-world scenario, you would load data (e.g., from a CSV).
    # Here, we create a synthetic dataset that simulates a stock price with an
    # upward trend and random volatility.
    # ----------------------------------------------------------------------
    time = np.arange(1, 101).reshape(-1, 1) # Time vector (e.g., 100 days)
    
    # y = mx + c, where m is the slope (trend) and c is the intercept
    actual_trend = 2 * time + 10
    noise = 10 * np.random.randn(len(time), 1) # Add random fluctuations
    data = actual_trend + noise

    # ----------------------------------------------------------------------
    # 2. Data Preparation: Split into Training and Testing Sets
    # We use the first 70% of the data to "train" our model and the last
    # 30% to "test" how well it predicts future values.
    # ----------------------------------------------------------------------
    train_size = int(0.7 * len(data))
    train_time = time[:train_size]
    train_data = data[:train_size]
    
    test_time = time[train_size:]
    test_data = data[train_size:] # This is the "actual" future data

    # ----------------------------------------------------------------------
    # 3. Quantitative Modeling: Linear Regression
    # We use Scikit-learn's LinearRegression to fit a line to our training data.
    # It automatically finds the best slope (m) and intercept (c).
    # ----------------------------------------------------------------------
    model = LinearRegression()
    model.fit(train_time, train_data)

    # Extract model coefficients
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Create the fitted line for visualization
    fitted_line = model.predict(train_time)

    # ----------------------------------------------------------------------
    # 4. Forecasting
    # Use the trained model to predict values for the "test" period.
    # ----------------------------------------------------------------------
    forecast_values = model.predict(test_time)

    # ----------------------------------------------------------------------
    # 5. Model Evaluation
    # Compare predicted values against actual values using Root Mean
    # Squared Error (RMSE). A lower RMSE means a better fit.
    # ----------------------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(test_data, forecast_values))
    
    # Display the results
    print('\n--- Quantitative Model Results ---')
    print(f'Linear Model Equation: y = {slope:.2f}*x + {intercept:.2f}')
    print(f'Forecasting Error (RMSE): {rmse:.2f}')
    print('----------------------------------\n')
    print('Displaying plot... Close the plot window to end the script.')

    # ----------------------------------------------------------------------
    # 6. Visualization
    # A plot is the best way to see how the model performed.
    # ----------------------------------------------------------------------
    plt.figure(figsize=(12, 7), facecolor='#f4f4f4')
    ax = plt.axes()
    ax.set_facecolor('#e6e6e6')
    
    # Plotting the data points
    plt.scatter(train_time, train_data, color='blue', label='Training Data', alpha=0.7)
    plt.scatter(test_time, test_data, color='black', label='Actual Future Data')
    
    # Plotting the model lines
    plt.plot(train_time, fitted_line, 'r-', linewidth=2, label='Fitted Trend Line')
    plt.plot(test_time, forecast_values, 'g--', linewidth=2, label='Forecast')
    
    # Adding labels and title for clarity
    plt.title('Time Series Forecasting using Linear Regression', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Value (e.g., Stock Price)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# This is the standard entry point for a Python script
if __name__ == "__main__":
    quantitative_modeling()

