import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Download stock data
stock = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Step 2: Prepare data
stock["Days"] = range(len(stock))
X = stock[["Days"]]
y = stock["Close"]

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict the next 30 days
future_days = pd.DataFrame({"Days": range(len(stock), len(stock) + 30)})
predicted_prices = model.predict(future_days)

# Step 5: Plot
plt.plot(stock["Days"], stock["Close"], label="Historical Prices")
plt.plot(future_days["Days"], predicted_prices, label="Predicted Prices")
plt.legend()
plt.show()
