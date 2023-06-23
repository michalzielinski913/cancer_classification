import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from plots import plot_confusion_matrix, plot_correlation_matrix


def prepare_data():
    # Prepare the dataset
    df = pd.read_csv("../Data/export.csv")

    encoder = LabelEncoder()
    df['histopathology'] = encoder.fit_transform(df['histopathology'])

    return df


def feature_selection(corr_matrix: pd.DataFrame, target_feature, min_correlation=0.16):
    plot_correlation_matrix(corr_matrix)

    # Select correlated features
    correlated_features = corr_matrix.index[corr_matrix[target_feature].abs() > min_correlation].tolist()

    correlated_features.remove(target_feature)
    return correlated_features


def train_classifier(train_df):
    X_train = train_df.filter(selected_features, axis=1)
    y_train = train_df["histopathology"]

    # Create naive bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    return naive_bayes


def validate(classifier, test_df):
    X_test = test_df.filter(selected_features, axis=1)
    y_test = test_df["histopathology"]

    # Create and display confusion matrix
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    # Oblicz dokładność klasyfikacji
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Dokładność: ", accuracy)


if __name__ == "__main__":
    df = prepare_data()

    train_df, test_df = train_test_split(df, train_size=0.8, random_state=42, stratify=df['histopathology'])

    selected_features = feature_selection(train_df.corr(), 'histopathology', 0.15)

    print('Selected features:', selected_features)

    classifier = train_classifier(train_df)
    validate(classifier, test_df)
