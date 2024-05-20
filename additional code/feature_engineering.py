import pandas as pd

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, low_limit, up_limit):
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe['Is_Outlier'] = 0

# Load dataset
df = pd.read_csv('final-project/data/processed/processed_data.csv')

# Drop unnecessary column
df = df.drop(['Adj Close'], axis=1)

# Split date into year, month, and day
df[['Year', 'Month', 'Day']] = df['Date'].str.split('-', expand=True).astype('int')

# Calculate outliers using IQR method
Q1 = df['Close'].quantile(0.25)
Q3 = df['Close'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Save the processed data with outliers
df['Is_Outlier'] = ((df['Close'] < lower_bound) | (df['Close'] > upper_bound)).astype(int)
df.to_csv('final-project/data/processed/processed_data_with_outliers.csv', index=False)

# Replace outliers
replace_with_thresholds(df, 'Close', lower_bound, upper_bound)

# Save the processed data without outliers
df.to_csv('final-project/data/processed/processed_data_without_outliers.csv', index=False)