import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif,f_regression
from sklearn.model_selection import train_test_split

def train_validate_split(df, target_col="histopathology", train_percent=.8, validate_percent=.2, seed=42):
    # Calculate the train and validate size
    train_size = train_percent / (train_percent + validate_percent)

    # Split the dataframe into training and validation sets, stratifying on the target column
    train_df, validate_df = train_test_split(df, train_size=train_size, random_state=seed, stratify=df[target_col])

    return train_df, validate_df

def select_k_best_features(df, k):
    # Separate the features and labels
    X = df.drop(columns=["histopathology"])
    y = df["histopathology"]

    # Apply SelectKBest with f_classif scoring function
    selector = SelectKBest(score_func=f_classif, k=k)

    # Fit the selector to the data
    selector.fit(X, y)

    # Get the names of the selected features
    mask = selector.get_support()
    feature_names = X.columns[mask]
    return feature_names


if __name__=="__main__":

    df=pd.read_csv("Data/cancer_cleaned.csv", sep=",")
    y = df["histopathology"]

    train_df, validate_df=train_validate_split(df)
    feature_names = select_k_best_features(train_df, k=16)
    df_new = pd.DataFrame(df, columns=feature_names)
    df_new["histopathology"] = y.values

    df_new.to_csv("Data/export.csv", index=False)