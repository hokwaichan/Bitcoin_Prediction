import pandas as pd

df = pd.read_csv('final-project/data/processed/test.csv')
df.drop(columns=['Open','High','Low','Close','Volume','Year','Month','Day','Is_Outlier'], inplace=True)
df.to_csv('final-project/data/processed/test_labels_baseline.csv', index=False)