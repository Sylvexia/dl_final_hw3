import torch
import torch.nn as nn

from data_proc import get_data, standardize_data, get_label_names, gen_confusion_matrix, get_class_report, gen_loss_plt
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pandas as pd
import os


class NN(nn.Module):
    def __init__(self, class_num):
        super(NN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(9, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 144),
            nn.ReLU(inplace=True),
            nn.Linear(144, 288),
            nn.ReLU(inplace=True),
            nn.Linear(288, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, class_num)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


data_frame = pd.read_csv('feature_time_48k_2048_load_1.csv')
train, test = get_data(data_frame)

train_label = train.iloc[:, -1].to_numpy()
test_label = test.iloc[:, -1].to_numpy()

train_data = train.iloc[:, :-1].to_numpy()
test_data = test.iloc[:, :-1].to_numpy()

train_data, test_data = standardize_data(train_data, test_data)

le = preprocessing.LabelEncoder()
train_label_encoded = le.fit_transform(train_label)

test_label_encoded = le.fit_transform(test_label)

BATCH_SIZE = 16
NUM_EPOCHS = 1000

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(
    train_data).float(), torch.from_numpy(train_label_encoded).long())
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(
    test_data).float(), torch.from_numpy(test_label_encoded).long())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True)

model = NN(10)
model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_y_true = []
train_y_pred = []
test_y_true = []
test_y_pred = []
train_losses = []
test_losses = []
dir = "neural_network"

for epoch in range(NUM_EPOCHS):
    model = model.train()
    cum_training_loss = 0.0
    for i, (data, label) in enumerate(train_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        
        cum_training_loss += loss.item()*data.size(0)
        _, pred = torch.max(output.data, 1)
        train_y_true += label.tolist()
        train_y_pred += pred.tolist()

    train_loss = cum_training_loss/len(train_loader)
    train_losses.append(train_loss)
    train_report = classification_report(
        y_true=train_y_true, y_pred=train_y_pred, output_dict=True)
    print(
        f'Epoch: {epoch}, Loss: {train_loss}, accuracy: {train_report["accuracy"]}')

    model = model.eval()
    best_test_accuracy = 0.0

    with torch.no_grad():
        cum_testing_losses = 0.0
        for i, (data, label) in enumerate(test_loader):
            data = data.to('cuda')
            label = label.to('cuda')

            output = model(data)
            loss = criterion(output, label)

            cum_testing_losses += loss.item()*data.size(0)
            _, pred = torch.max(output.data, 1)
            test_y_true += label.tolist()
            test_y_pred += pred.tolist()

        test_loss = cum_testing_losses/len(test_loader)
        test_losses.append(test_loss)
        test_report = classification_report(
            y_true=test_y_true, y_pred=test_y_pred, output_dict=True)
        accuracy = test_report["accuracy"]
        print(f'Epoch: {epoch}, Loss: {test_loss}, accuracy: {accuracy}')

        if accuracy > best_test_accuracy:
            if(not os.path.exists(dir)):
                os.makedirs(dir)
            torch.save(model.state_dict(), f'{dir}/best_model.pth')

best_model = NN(10)
best_model = best_model.to('cuda')
best_model.load_state_dict(torch.load(f'{dir}/best_model.pth'))

res_test_true = []
res_test_pred = []
res_train_true = []
res_train_pred = []

with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        output = best_model(data)
        _, pred = torch.max(output.data, 1)
        res_test_true += label.tolist()
        res_test_pred += pred.tolist()

    for i, (data, label) in enumerate(train_loader):
        data = data.to('cuda')
        label = label.to('cuda')

        output = best_model(data)
        _, pred = torch.max(output.data, 1)
        res_train_true += label.tolist()
        res_train_pred += pred.tolist()

res_true = le.inverse_transform(res_test_true)
res_pred = le.inverse_transform(res_test_pred)
train_true = le.inverse_transform(res_train_true)
train_pred = le.inverse_transform(res_train_pred)

get_class_report(train_pred, train_true, res_pred, res_true, dir)

gen_confusion_matrix(train_true, train_pred, res_true, res_pred,
                     get_label_names(data_frame), "NeuralNetwork", dir)

gen_loss_plt(train_losses, test_losses, dir)

# predict_label = [0, 3, 1, 5, 3, 6, 7]
# print(le.inverse_transform(predict_label))
