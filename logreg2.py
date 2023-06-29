from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scipy.stats as ss
import seaborn as sns

def plot_correlation_matrix(corr_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()

def feature_selection(corr_matrix: pd.DataFrame, target_feature, min_correlation):
    plot_correlation_matrix(corr_matrix)

    correlated_features = corr_matrix.index[corr_matrix[target_feature].abs() > min_correlation].tolist()

    correlated_features.remove(target_feature)
    return correlated_features

df = pd.read_csv("Data/cancer_cleaned.csv")

le = LabelEncoder()
df["histopathology"] = le.fit_transform(df["histopathology"])


X = df.drop(columns=["histopathology"])
y = df["histopathology"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
corr_matrix = train_df.corr()

correlated_features = feature_selection(corr_matrix, target_feature="histopathology", min_correlation=0.06)

selected_features_df = train_df[correlated_features]
print(selected_features_df)

X_train = selected_features_df
X_test = X_test[correlated_features]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=100)

cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Accuracy:", cv_scores.mean())

param_grid = {"C": [0.1, 0.2, 1.0, 10.0, 25.0, 100.0],
              "penalty": ['l1', 'l2', 'none'],
              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag'],
              'multi_class' : ['multinomial']
              }

grid_search = GridSearchCV(model, param_grid = param_grid, cv=3, verbose=True, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
print(best_model)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
labels = best_model.classes_
labels = le.inverse_transform(labels)
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]  

plt.figure(figsize=(8, 6))
plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

thresh = cm.max() / 2.
for i in range(len(labels)):
    for j in range(len(labels)):
         plt.text(j, i, format(cm[i, j], 'd') + '\n({:.2%})'.format(cm_percent[i, j]), 
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

plt.tight_layout()
plt.show()