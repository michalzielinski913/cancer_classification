import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Prepare the dataset
df = pd.read_csv("Data/export.csv")

X = df.drop(columns=["histopathology"])
y = df["histopathology"]
report_data = []

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
        n_jobs=-2,
        class_weight={'SQUAMOUS': 1, 'OTHER': 2.2}
    )
    score=np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))
    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)

    # Calculate the accuracy on the validation data
    val_accuracy = accuracy_score(y_val, y_pred_val)

    # Append tested values, training score, and validation accuracy to report_data
    report_data.append({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'criterion': criterion,
        'training_score': score,
        'validation_accuracy': val_accuracy
    })

    return score
    return score

# Create an Optuna study with a fixed number of iterations (trials)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=1000)

# Print the best parameters found
print("Best parameters found: ", study.best_params)

# Evaluate the optimized model on the validation set
best_clf = RandomForestClassifier(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        bootstrap=study.best_params['bootstrap'],
        criterion=study.best_params['criterion'],
        random_state=42,
        n_jobs=-2,
        class_weight={'SQUAMOUS': 1, 'OTHER': 2.2}
)

best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_val)

# Get predicted probabilities for each class
y_pred_prob = best_clf.predict_proba(X_val)

# Get class labels from the trained classifier
class_labels = best_clf.classes_

# Reset the index of y_val before creating the validation_results DataFrame
y_val_reset = y_val.reset_index(drop=True)

# Save the validation results as a CSV file
validation_results = pd.DataFrame({'Real Values': y_val_reset,
                                   'Predicted Values': y_pred,
                                   f'Confidence for {class_labels[0]}': y_pred_prob[:, 0],
                                   f'Confidence for {class_labels[1]}': y_pred_prob[:, 1]})

validation_results.to_csv('Results/validation_results.csv', index=False)
report_df = pd.DataFrame(report_data)
report_df.to_csv('Results/report.csv', index=False)