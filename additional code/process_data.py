import pandas as pd

# Read the dataset
data = pd.read_csv('final-project/data/raw/BTC-USD.csv')

# Calculate the buy/not buy column
# If the closing price of tomorrow is expected to be higher than today's closing price, we'll mark it as a "Buy" (1).
# If the closing price of tomorrow is expected to be lower than today's closing price, we'll mark it as "Not Buy" (0).
data['Buy_or_Not'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Replace the last entry with NaN as there's no information about the next day's closing price
data.iloc[-1, -1] = pd.NA

# Save the updated dataset
data.to_csv('final-project/data/processed/processed_data.csv', index=False)