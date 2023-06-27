import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


def feature_selection(X, y, n_features):
    
    # Feature selection using SelectKBest and f_classif scoring function
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
   
    selected_features = X.columns[selector.get_support()]

    return selected_features

def train_classifier(X_train, X_test, y_train, y_test):
    # SVM model
    svm_model = SVC(kernel='sigmoid', gamma='scale', C=3)
    svm_model.fit(X_train, y_train)

    # Prediction
    y_pred = svm_model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_plot(cm)

    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Cross-validation
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", cv_scores.mean())

def confusion_matrix_plot(matrix):
    total_samples = matrix.sum(axis=1, keepdims=True)

    percentage_matrix = (matrix / total_samples) * 100

    class_names = ['OTHER', 'SQUAMOUS']
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=False, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Adding the counts on top of the percentages
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = f'{percentage_matrix[i, j]:.2f}%'
            plt.text(j + 0.5, i + 0.5, f'{matrix[i, j]}\n\n{percentage}', ha='center', va='center')

    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

# Dataset preparation
df = pd.read_csv("Data/export.csv")

encoder = LabelEncoder()
df['histopathology'] = encoder.fit_transform(df['histopathology'])

X = df.drop(columns=["histopathology"])
y = df["histopathology"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42, stratify=df['histopathology'])

# Perform feature selection on the training data only
selected_features = feature_selection(X_train, y_train, n_features=5)  # Change the value of n_features as desired

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print('Selected features:', X_train.columns)

train_classifier(X_train, X_test, y_train, y_test)
