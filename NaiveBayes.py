import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


def feature_selection(corr_matrix: pd.DataFrame, target_feature, min_correlation=0.16):
    # Display correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()

    # Select correlated features
    correlated_features = corr_matrix.index[corr_matrix[target_feature].abs() > min_correlation].tolist()

    return correlated_features


def train_classifier():
    # Create naive bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    # Create and display confusion matrix
    y_pred = naive_bayes.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# Prepare the dataset
df = pd.read_csv("Data/export.csv")

encoder = LabelEncoder()
df['histopathology'] = encoder.fit_transform(df['histopathology'])

selected_features = feature_selection(df.corr(), 'histopathology', 0.16)

X = df.filter(selected_features, axis=1).drop(columns=["histopathology"])
y = df["histopathology"]
print('Selected features:', X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_classifier()
