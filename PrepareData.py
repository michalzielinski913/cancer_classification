import pandas as pd
from feature import train_validate_split, select_k_best_features
from tqdm import tqdm

df = pd.read_csv("Data/cancer_cleaned.csv", sep=",")
y = df["histopathology"]
train_df, validate_df = train_validate_split(df)
for i in tqdm(range(1,32)):
    feature_names = select_k_best_features(train_df, k=i)

    df_train = pd.DataFrame(train_df, columns=feature_names)
    df_train["histopathology"] = train_df['histopathology'].values
    df_train.to_csv("Data/Input/train_{}_features.csv".format(i), index=False)

    df_validate = pd.DataFrame(validate_df, columns=feature_names)
    df_validate["histopathology"] = validate_df['histopathology'].values
    df_validate.to_csv("Data/Input/validate_{}_features.csv".format(i), index=False)