import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('final-project/data/raw/BTC-USD.csv')

data['Date'] = pd.to_datetime(data['Date'])

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(data['Date'], data['Close'])
plt.title('Bitcoin Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.xticks(rotation=45)
plt.xlim(data['Date'].min(), data['Date'].max())
plt.tight_layout()
plt.show()
