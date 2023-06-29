#Basic utility modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna

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
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

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
    def __init__(self, n_features, n_neurons):
        super(BinaryClassification, self).__init__()
        # Number of input features is n
        # half_neurons = (int)n_neurons/2
        self.layer_1 = nn.Linear(n_features, n_neurons) 
        self.layer_2 = nn.Linear(n_neurons, n_neurons)
        self.layer_out = nn.Linear(n_neurons, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_neurons)
        self.batchnorm2 = nn.BatchNorm1d(n_neurons/2)
        
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([class_weights]), reduction='mean')
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # criterion= nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    # loss_weighted = criterion_weighted(x, y)

    n_epochs = 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

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
        # print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        if early_stopper.early_stop(epoch_loss):             
            break
    return (epoch_acc/len(train_loader))
            

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




def objective(trial):
    n_features = trial.suggest_int('n_features', 1, 15)
    class_weights = trial.suggest_float('class_weights', 0.001, 10)
    n_neurons = trial.suggest_int('n_neurons', 10, 200)
    batch_size = trial.suggest_int('batch_size', 5, 50)
    
    X = df.drop(columns=["histopathology"])
    X = df.drop(df.iloc[:, n_features:], axis=1)
    y = df["histopathology"]
    mapa = {'SQUAMOUS': 1, 'OTHER': 0}
    y = y.map(mapa)

    print(X.shape)
    # print(y.shape)
    # print(y.value_counts())

    # print(y)

    # Split the dataset into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), 
                        torch.FloatTensor(y_train.to_numpy()))

    test_data = TestData(torch.FloatTensor(X_val.to_numpy()))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = BinaryClassification(n_features, n_neurons)

    acc = trainingLoop(train_loader, model, class_weights)
    preds = test_model(test_loader, model)
    score = binary_acc(torch.FloatTensor(np.array(preds)), torch.FloatTensor(y_val.to_numpy()))
    saved_models.append(model)
    report_data.append({
        'n_features': n_features,
        'n_neurons': n_neurons,
        'class_weights': class_weights,
        'batch_size': batch_size,
        'train_accuracy': acc,
        'score': score,
    })

    return score


if __name__ == '__main__':
    report_data = []
    #models are very small so they can be kept in the memory
    saved_models = []
    # Prepare the dataset
    df = pd.read_csv("Data/export.csv")

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    # Print the best parameters found
    print("Best parameters found: ", study.best_params)
    best_model = saved_models[study.best_trial.number]

    X = df.drop(columns=["histopathology"])
    X = df.drop(df.iloc[:, study.best_trial.params['n_features']:], axis=1)
    y = df["histopathology"]
    mapa = {'SQUAMOUS': 1, 'OTHER': 0}
    y = y.map(mapa)

    print(X.shape)
    # print(y.shape)
    # print(y.value_counts())

    # print(y)

    # Split the dataset into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), 
                        torch.FloatTensor(y_train.to_numpy()))

    train_data_flat = TestData(torch.FloatTensor(X_train.to_numpy()))
    train_loader_flat = DataLoader(dataset=train_data_flat, batch_size=1)

    test_data = TestData(torch.FloatTensor(X_val.to_numpy()))

    train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    preds_val = test_model(test_loader, best_model)
    preds_train = test_model(train_loader_flat, best_model)

    #confusion matrix
    cm = confusion_matrix(y_val, preds_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix_val.jpg")
    # plt.show()

    cm = confusion_matrix(y_train, preds_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix_train.jpg")

    fpr, tpr, _ = roc_curve(y_val, preds_val)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig("rog_curve_val.jpg")

    fpr, tpr, _ = roc_curve(y_train, preds_train)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig("rog_curve_train.jpg")

    prec, recall, _ = precision_recall_curve(y_val, preds_val)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.savefig("recall_curve_val.jpg")

    prec, recall, _ = precision_recall_curve(y_train, preds_train)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.savefig("recall_curve_train.jpg")

    plt.show()

    #save the model
    torch.save(best_model.state_dict(), "./best_model.pth")