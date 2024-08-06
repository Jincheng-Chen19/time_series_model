import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


os.chdir('E:/work/seq_env/lucky/random/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


def create_dataset(dataset, result):
    X, y = [], []
    for i in range(dataset.shape[0]):
        Xi = []
        for j in range(dataset.shape[1]):
            Xi.append([dataset[i, j]])
        X.append(Xi)
        y.append([result[i]])
    return torch.tensor(X).to(device), torch.tensor(y).to(device)


def split_train(X, y, n_epochs=100, batch_size=10, test_size=0.2, valid_size=0.25, learning_rate=0.01):
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv, test_size=valid_size, random_state=0)
    X_train, y_train = create_dataset(X_train, y_train)
    X_valid, y_valid = create_dataset(X_valid, y_valid)
    X_test, y_test = create_dataset(X_test, y_test)
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False)
    prev_loss = np.inf
    model = RandomModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    epoch_final = 0
    train_losses = np.zeros(n_epochs)
    valid_losses = np.zeros(n_epochs)
    test_loss = 0
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0.0
        train_number = 0
        for batch, [X_batch, y_batch] in enumerate(train_loader):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_number += y_batch.size(0)
        train_losses[epoch] = train_loss / train_number
        print(f"epoch:{epoch + 1}, train_loss:{train_losses[epoch]}")
        model.eval()
        valid_loss = 0.0
        valid_number = 0
        with torch.no_grad():
            for batch, [X_batch, y_batch] in enumerate(valid_loader):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                valid_loss += loss.item()
                valid_number += y_batch.size(0)
        valid_losses[epoch] = valid_loss / valid_number
        print(f"epoch:{epoch + 1}, valid_loss:{valid_losses[epoch]}")
        final_loss = (valid_losses[epoch] + train_losses[epoch]) / 2
        if final_loss < prev_loss:
            prev_loss = loss
            torch.save(model.state_dict(), './models/random_model.pt') # change to a proper path in colab
            epoch_final = epoch
    print(f"After epoch: {epoch_final + 1}, the model showed best performance.\n "
          f"Load the parameter generated after this epoch")
    plt.plot(x=range(n_epochs), y=train_losses, label='train loss', color='blue')
    plt.plot(x=range(n_epochs), y=valid_losses, label='valid loss', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    model.load_state_dict(torch.load('./models/random_model.pt'))
    model.eval()
    test_loss = 0.0
    test_number = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
            test_number = y_batch.size(0)
    final_test_loss = test_loss / test_number
    print(f"test_loss: {final_test_loss}")
    return model


def k_fold_train(X, y, n_split=10, n_epochs=100, batch_size=10, learning_rate=0.01):
    kf = KFold(n_splits=n_split, shuffle=True, random_state=0)
    train_losses = np.zeros([n_split, n_epochs])
    test_losses = np.zeros(n_split)
    for i, [train_index, test_index] in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = create_dataset(X_train, y_train)
        X_test, y_test = create_dataset(X_test, y_test)
        train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False)
        model = RandomModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss().to(device)
        epoch_final = 0
        train_loss_k = np.zeros(n_epochs)
        for epoch in tqdm(range(n_epochs)):
            model.train()
            train_loss = 0.0
            train_number = 0
            for batch, [X_batch, y_batch] in enumerate(train_loader):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_number += y_batch.size(0)
            train_loss_k[epoch] = train_loss / train_number
            print(f"epoch:{epoch + 1}, train_loss:{train_loss_k[epoch]}")
            if train_loss_k[epoch] < prev_loss:
                prev_loss = train_loss_k[epoch]
                torch.save(model.state_dict(), f'./models/random_model_{i}.pt')
                epoch_final = epoch
        print(f"After epoch: {epoch_final + 1}, the model showed best performance.\n "
              f"Load the parameter generated after this epoch")
        plt.plot(x=range(n_epochs), y=train_loss_k, label='train loss', color='blue')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        model.load_state_dict(torch.load(f'./models/random_model_{i}.pt'))
        model.eval()
        test_loss = 0.0
        test_number = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
                test_number = y_batch.size(0)
        final_test_loss = test_loss / test_number
        print(f"test_loss: {final_test_loss}")
        train_losses[i] = train_loss_k
        test_losses[i] = final_test_loss
    print(f'After {n_split}-fold cross validation, the mean test loss is {np.mean(test_losses)}')
    mean_train_loss = np.mean(train_losses, axis=0)
    plt.plot(mean_train_loss, label='mean train loss', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


dataset = pd.read_csv('./data/LSTM_input.csv', sep=',')
colname = dataset.loc[:, 'mask'].astype(str)
dataset = dataset.drop(['mask'], axis=1)
dataset = dataset.transpose()
dataset.columns = colname
dataset = np.nan_to_num(dataset, nan=0)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
result = pd.read_csv('./data/LSTM_output.csv')
X = scaled_data.transpose().astype('float32')

y = np.array(result.loc[:, 'result'].to_list()).astype('float32')


model = RandomModel().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss().to(device)

