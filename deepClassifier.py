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
from sklearn.metrics import confusion_matrix, classification_report

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 16
        self.layer_1 = nn.Linear(16, 10) 
        self.layer_2 = nn.Linear(10, 10)
        self.layer_out = nn.Linear(10, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(10)
        self.batchnorm2 = nn.BatchNorm1d(10)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

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

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def trainingLoop(train_loader, model):


    learning_rate = 1e-2

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for features, label in train_loader:
            label = label.unsqueeze(1)
            # features, label = features.to(device), label.to(device)
            
            # print(len(features))
            # print(len(label))
            # print(features.shape)
            # print(label.shape)
            # print(type(features))
            # print(type(label))
            features = features.to(torch.float32)
            outputs = model(features)

            loss = criterion(outputs, label)
            acc = binary_acc(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

def test_model(test_loader, model):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            # X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return y_pred_list



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

train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), 
                       torch.FloatTensor(y_train.to_numpy()))

test_data = TestData(torch.FloatTensor(X_val.to_numpy()))

train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

model = BinaryClassification()

trainingLoop(train_loader, model)
preds = test_model(test_loader, model)

print(confusion_matrix(y_val, preds))
# print(classification_report(y_val, preds))