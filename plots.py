import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

amount_of_results=31
data_frames=[]
for i in range(1, amount_of_results+1):
    data_frame=pd.read_csv("Results/report_{}.csv".format(i))
    data_frames.append(data_frame)
print(len(data_frames))

training_scores_max = [df['training_score'].max() for df in data_frames]
validation_accuracy_max = [df['validation_accuracy'].max() for df in data_frames]
index = list(range(1, len(data_frames) + 1))

plt.figure(figsize=(12,6))

# Plot training_scores_max
plt.plot(index, training_scores_max, label='Max Training Accuracy')

# Plot validation_accuracy_max
plt.plot(index, validation_accuracy_max, label='Max Validation Accuracy')

plt.xlabel('Amount of features')
plt.ylabel('Max Accuracy')
plt.title('Accuracy vs amount of features')
plt.xticks(index)

plt.legend()
plt.show()