#Basic utility modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#PyTorch
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Utilities from sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_train_loss = np.inf

    def early_stop(self, train_loss):
        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
            self.counter = 0
        elif train_loss > (self.min_train_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class BinaryClassification(nn.Module):
    def __init__(self, n_features):
        super(BinaryClassification, self).__init__()
        # Number of input features is n
        self.layer_1 = nn.Linear(n_features, 20) 
        self.layer_2 = nn.Linear(20, 20)
        self.layer_out = nn.Linear(20, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(20)
        self.batchnorm2 = nn.BatchNorm1d(20)
        
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

def trainingLoop(train_loader, model, class_weights):


    learning_rate = 1e-3

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.65]), reduction='mean')
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # criterion= nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    # loss_weighted = criterion_weighted(x, y)

    n_epochs = 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for features, label in train_loader:
            label = label.unsqueeze(1)
            # features, label = features.to(device), label.to(device)
            
            features = features.to(torch.float32)
            outputs = model(features)
            # print(label.shape)
            # print(outputs.shape)
            loss = criterion(outputs, label)
            acc = binary_acc(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        if early_stopper.early_stop(epoch_loss):             
            break
            

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

# X = df.drop(columns=["histopathology"])
X = df.drop(df.iloc[:, 10:], axis=1)
y = df["histopathology"]
mapa = {'SQUAMOUS': 1, 'OTHER': 0}
y = y.map(mapa)

print(X.shape)
print(y.shape)
print(y.value_counts())

print(y)

# Split the dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), 
                       torch.FloatTensor(y_train.to_numpy()))

test_data = TestData(torch.FloatTensor(X_val.to_numpy()))

train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y = y_train.to_numpy())
class_weights=torch.tensor(class_weights,dtype=torch.float)
print(class_weights)

model = BinaryClassification(10)

trainingLoop(train_loader, model, class_weights)
preds = test_model(test_loader, model)

cm = confusion_matrix(y_val, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()