import time
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


df=pd.read_csv("../cancer_classification/cancer_cleaned.csv", sep=";")
y=df['histopathology'].values
x=df[df.columns[2:]].to_numpy()

feature_names= (df.columns.to_numpy())[2:]
feature_names=[feature.replace("original_", "") for feature in feature_names]

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)


start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
import matplotlib.pyplot as plt
forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots(figsize=(20, 10))
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
res=[]
for i, feature in enumerate(result.importances_mean):
    if feature>0.01:
        res.append(i+2)
print(len(res))

x_extract=df[df.columns[res]]
print(x_extract.head())

x_extract[df['histopathology'].name] = df['histopathology']

x_extract.to_csv("export.csv", index=False)