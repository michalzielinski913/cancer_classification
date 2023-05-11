#Basic utility modules
import numpy as np
import pandas as pd

#PyTorch
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Utilities from sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def trainingLoop(train_loader):
    model = nn.Sequential(
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.Softmax(dim=1)
    )

    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    loss_fn = nn.NLLLoss()

    n_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(n_epochs):
        for features, label in train_loader:
            # features, label = features.to(device), label.to(device)

            features = features.to(torch.float32)
            outputs = model(features)
            loss = loss_fn(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))



# Prepare the dataset
df = pd.read_csv("Data/export.csv")

X = df.drop(columns=["histopathology"])
y = df["histopathology"]
mapa = {'SQUAMOUS': 1, 'OTHER': 0}
y = y.map(mapa)

print(X.shape)
print(y.shape)

# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = TrainData(torch.LongTensor(X_train.to_numpy()), 
                       torch.LongTensor(y_train.to_numpy()))

test_data = TestData(torch.LongTensor(X_val.to_numpy()))

train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

trainingLoop(train_loader)