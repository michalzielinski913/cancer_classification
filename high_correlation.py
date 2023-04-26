
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif,f_regression
def balance_classes(df, label_col):
    # Get the counts for each class label
    class_counts = df[label_col].value_counts()

    # Compute the target number of samples for each class label
    target_count = class_counts.min()

    # Create a list of indices for each class label
    indices_by_label = [df[df[label_col] == label].index.tolist() for label in class_counts.index]

    # Randomly select a subset of rows for each class label
    new_indices = []
    for indices in indices_by_label:
        new_indices.extend(np.random.choice(indices, size=target_count, replace=False))

    # Create a new DataFrame with the balanced class distribution
    df_balanced = df.loc[new_indices]
    return df_balanced
def select_k_best_features(df, k):
    # Separate the features and labels
    X = df.drop(columns=["histopathology"])
    y = df["histopathology"]

    # Apply SelectKBest with f_classif scoring function
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the names of the selected features
    mask = selector.get_support()
    feature_names = X.columns[mask]

    # Create a new DataFrame with the selected features and labels
    df_new = pd.DataFrame(X_new, columns=feature_names)
    df_new["histopathology"] = y.values
    return df_new
df=pd.read_csv("cancer_cleaned.csv", sep=",")



# Remove highly correlated features with a threshold of 0.9
df=balance_classes(df, "histopathology")
print(df['histopathology'].value_counts())

df_reduced = select_k_best_features(df, k=16)

# Add the label column back to the reduced DataFrame
print(df_reduced)
df_reduced.to_csv("export.csv", index=False)