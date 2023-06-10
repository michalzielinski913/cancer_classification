import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    y_pred_val = clf.predict(X_validation)

    # Calculate the accuracy on the validation data
    val_accuracy = accuracy_score(y_validation, y_pred_val)

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

for i in tqdm(range(1,32)):

    train_df=pd.read_csv("Data/Input/train_{}_features.csv".format(i), sep=",")
    X_train = train_df.drop(columns=["histopathology"])
    y_train = train_df["histopathology"]

    validation_df=pd.read_csv("Data/Input/validate_{}_features.csv".format(i), sep=",")
    X_validation = validation_df.drop(columns=["histopathology"])
    y_validation = validation_df["histopathology"]

    report_data = []

    # Create an Optuna study with a fixed number of iterations (trials)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)
    # Print the best parameters found
    print("Best parameters found: ", study.best_params)

    report_df = pd.DataFrame(report_data)
    report_df.to_csv('Results/report_{}.csv'.format(i), index=False)