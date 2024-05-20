import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

# Define hyperparameters grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Build the model (Logistic Regression)
model = LogisticRegression(random_state=20)

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=5)
grid_search.fit(x_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print('Best model:', best_model)

# Make predictions on the validation set
val_predictions_probs = best_model.predict_proba(x_val_scaled)[:, 1]

# Evaluate the model on the validation set
auroc_val = sklearn.metrics.roc_auc_score(y_val, val_predictions_probs)
print(f'AUROC on validation set: {auroc_val}')

# Plot ROC curve
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_val, val_predictions_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auroc_val)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Validation Set')
plt.legend(loc='lower right')
plt.show()

# Calculate AUROC on the training set
train_predictions_probs = best_model.predict_proba(x_train_scaled)[:, 1]
auroc_train = sklearn.metrics.roc_auc_score(y_train, train_predictions_probs)
print(f'AUROC on training set: {auroc_train}')

# Load test data.
df_test = pd.read_csv('final-project/data/processed/test.csv')

# Handle missing values
df_test.dropna(inplace=True)

# Preprocess the test data
x_test = df_test[features]
x_test_scaled = scaler.transform(x_test)

# Make predictions on the test set
test_predictions_probs = best_model.predict_proba(x_test_scaled)[:, 1]

# Calculate AUROC on the test set
auroc_test = sklearn.metrics.roc_auc_score(df_test['Buy_or_Not'], test_predictions_probs)
print(f'AUROC on test set: {auroc_test}')

# Create a DataFrame for submission
df_predictions = pd.DataFrame({'Date': df_test['Date'], 'Buy_or_Not': test_predictions_probs})

# Write predictions to file.
df_predictions.to_csv('final-project/models/Logistic_Regression/predictions.csv', index=False)
