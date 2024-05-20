import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load Training data
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

# Build and train the models
models = {
    "Logistic Regression": LogisticRegression(random_state=20),
    "Random Forest": RandomForestClassifier(random_state=20),
    "Gradient Boosting": GradientBoostingClassifier(random_state=20),
    "SVM": SVC(probability=True, random_state=20),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

ensemble_val_predictions_probs = np.zeros_like(y_val, dtype=float)

for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    val_predictions_probs = model.predict_proba(x_val_scaled)[:, 1]
    ensemble_val_predictions_probs += val_predictions_probs

# Average predictions for the ensemble
ensemble_val_predictions_probs /= len(models)

# Evaluate the ensemble model on the validation set
auroc_ensemble_val = sklearn.metrics.roc_auc_score(y_val, ensemble_val_predictions_probs)
print(f'AUROC on validation set using ensemble: {auroc_ensemble_val}')

# Plot ROC curve for the ensemble
fpr_ensemble, tpr_ensemble, thresholds_ensemble = sklearn.metrics.roc_curve(y_val, ensemble_val_predictions_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr_ensemble, tpr_ensemble, color='blue', label='ROC curve (AUC = %0.2f)' % auroc_ensemble_val)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Ensemble Model (Validation Set)')
plt.legend(loc='lower right')
plt.show()

# Load test data
df_test = pd.read_csv('final-project/data/processed/test.csv')

# Handle missing values
df_test.dropna(inplace=True)

# Preprocess the test data
x_test = df_test[features]
x_test_scaled = scaler.transform(x_test)

ensemble_test_predictions_probs = np.zeros(len(x_test), dtype=float)

for name, model in models.items():
    test_predictions_probs = model.predict_proba(x_test_scaled)[:, 1]
    ensemble_test_predictions_probs += test_predictions_probs

# Average predictions for the ensemble
ensemble_test_predictions_probs /= len(models)

# Evaluate the ensemble model on the test set
auroc_ensemble_test = sklearn.metrics.roc_auc_score(df_test['Buy_or_Not'], ensemble_test_predictions_probs)
print(f'AUROC on test set using ensemble: {auroc_ensemble_test}')

# Create a DataFrame for submission
df_ensemble_predictions = pd.DataFrame({'Date': df_test['Date'], 'Buy_or_Not': ensemble_test_predictions_probs})

# Write ensemble predictions to file
df_ensemble_predictions.to_csv('final-project/models/Ensemble/predictions.csv', index=False)
