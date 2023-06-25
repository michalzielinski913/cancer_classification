import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the preprocessed dataset
df = pd.read_csv("Data/export.csv")

# Separate the features and labels
X = df.drop(columns=["histopathology"])
y = df["histopathology"]

selected_features = feature_selection(X, y, n_features=5)  # Change the value of n_features as desired

X = df[selected_features]
y = df["histopathology"]
print('Selected features:', X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42, stratify=df['histopathology'])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm = SVC()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 'scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_svm = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm.predict(X_test_scaled)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(report)

# Print the best parameters found by grid search
print("Best Parameters:", grid_search.best_params_)
