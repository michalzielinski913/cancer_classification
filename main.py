import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Prepare the dataset
df = pd.read_csv("export.csv")

X = df.drop(columns=["histopathology"])
y = df["histopathology"]

# Split the dataset into train and validation sets
# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the objective function for the study
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=42,
        n_jobs=-2
    )

    return np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))

# Create an Optuna study with a fixed number of iterations (trials)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=1000)

# Print the best parameters found
print("Best parameters found: ", study.best_params)

# Evaluate the optimized model on the validation set
best_clf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_val)

# Print the classification report
from sklearn.metrics import classification_report
print("\nClassification report on validation set:\n", classification_report(y_val, y_pred))