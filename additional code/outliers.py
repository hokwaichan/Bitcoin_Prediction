import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('final-project/data/processed/processed_data_with_outliers.csv')

# Grouping by Year and Calculating means
data_grouped = df.groupby('Year').mean()
plt.figure(figsize=(10, 5))
data_grouped['Close'].plot.bar()
plt.title('Mean Close Price by Year')
plt.xlabel('Year')
plt.ylabel('Mean Close Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box plot to disply outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df['Close'], orient='h')
plt.title('Boxplot of Close Price')
plt.show()
