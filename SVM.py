import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


def feature_selection(X, y, n_features):
    
    # Perform feature selection using SelectKBest and f_classif scoring function
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)

    # Get the selected features
    
    selected_features = X.columns[selector.get_support()]

    return selected_features

def train_classifier(X_train, X_test, y_train, y_test):
    # Create SVM model
    svm_model = SVC()

    # Fit the model to the training data
    svm_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_model.predict(X_test)

    # Create and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)

    # Perform cross-validation
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())


# Prepare the dataset
df = pd.read_csv("Data/export.csv")

encoder = LabelEncoder()
df['histopathology'] = encoder.fit_transform(df['histopathology'])

X = df.drop(columns=["histopathology"])
y = df["histopathology"]

selected_features = feature_selection(X, y, n_features=5)  # Change the value of n_features as desired

X = df[selected_features]
y = df["histopathology"]
print('Selected features:', X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_classifier(X_train, X_test, y_train, y_test)
