import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(corr_matrix):
    # Display correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Macierz korelacji')
    plt.show()


def plot_confusion_matrix(matrix):
    total_samples = matrix.sum(axis=1, keepdims=True)

    percentage_matrix = (matrix / total_samples) * 100

    class_names = ['OTHER', 'SQUAMOUS']
    # Use seaborn to make the confusion matrix more legible
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
