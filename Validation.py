from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

generation_id=3

df=pd.read_csv("Results/report_{}.csv".format(generation_id), index_col=False)

train_df = pd.read_csv("Data/Input/train_{}_features.csv".format(generation_id), sep=",")
X_train = train_df.drop(columns=["histopathology"])
y_train = train_df["histopathology"]

validation_df = pd.read_csv("Data/Input/validate_{}_features.csv".format(generation_id), sep=",")
X_validation = validation_df.drop(columns=["histopathology"])
y_validation = validation_df["histopathology"]
row=df[df.validation_accuracy == df.validation_accuracy.max()]

clf = RandomForestClassifier(
    n_estimators=row["n_estimators"].values[0],
    max_depth=row["max_depth"].values[0],
    min_samples_split=row["min_samples_split"].values[0],
    min_samples_leaf=row["min_samples_leaf"].values[0],
    max_features=row["max_features"].values[0],
    bootstrap=row["bootstrap"].values[0],
    criterion=row["criterion"].values[0],
    random_state=42,
    n_jobs=-2,
    class_weight={'SQUAMOUS': 1, 'OTHER': 2.2}
)
clf.fit(X_train, y_train)

# Make predictions with the model
y_pred = clf.predict(X_validation)

# Create a confusion matrix
cm = confusion_matrix(y_validation, y_pred)
total_samples = cm.sum(axis=1, keepdims=True)

percentage_matrix = (cm / total_samples) * 100

class_names = ['OTHER', 'SQUAMOUS']
# Use seaborn to make the confusion matrix more legible
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=False, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Adding the counts on top of the percentages
for i in range(len(class_names)):
    for j in range(len(class_names)):
        percentage = f'{percentage_matrix[i, j]:.2f}%'
        plt.text(j+0.5, i+0.5, f'{cm[i, j]}\n\n{percentage}', ha = 'center', va = 'center')

plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()