import pandas as pd

# Load dataset
df = pd.read_csv('final-project/data/raw/BTC-USD.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Print missing values
print("Missing values in the dataset:")
print(missing_values)
print()
