import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load Training data.
df = pd.read_csv('final-project/data/processed/train.csv')
y = df['Buy_or_Not']

# Features to use for training
features = ['Open', 'High', 'Low', 'Close', 'Volume']
x = df[features]

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=20)

# Preprocess the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

# Build and train the model (XGBoost)
model = XGBClassifier(random_state=20)
model.fit(x_train_scaled, y_train)

# Make predictions on the validation set
val_predictions_probs = model.predict_proba(x_val_scaled)[:, 1]

# Evaluate the model on the validation set
auroc_val = sklearn.metrics.roc_auc_score(y_val, val_predictions_probs)
print(f'AUROC on validation set: {auroc_val}')

# Calculate AUROC on the training set
train_predictions_probs = model.predict_proba(x_train_scaled)[:, 1]
auroc_train = sklearn.metrics.roc_auc_score(y_train, train_predictions_probs)
print(f'AUROC on training set: {auroc_train}')

# Load test data.
df_test = pd.read_csv('final-project/data/processed/test.csv')

# Preprocess the test data
x_test = df_test[features]
x_test_scaled = scaler.transform(x_test)

# Make predictions on the test set
test_predictions_probs = model.predict_proba(x_test_scaled)[:, 1]

# Create a DataFrame for submission
df_predictions = pd.DataFrame({'Date': df_test['Date'], 'Buy_or_Not': test_predictions_probs})

# Write predictions to file.
df_predictions.to_csv('final-project/models/XGB/predictions.csv', index=False)