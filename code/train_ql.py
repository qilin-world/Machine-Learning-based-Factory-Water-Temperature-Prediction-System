import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import argparse
import matplotlib.pyplot as plt
from data_preprocessing import load_data, TimeSeriesPredictionDataset
from model import SimpleLSTM, TransformerModel
from torch.optim.lr_scheduler import StepLR


# 定义站点集合
Stations = ['s1', 's4', 's5', 's8', 's10', 's12', 's14', 's15']



# 参数类
class Parameters:
    def __init__(self):
        self.batch_size = 50
        self.input_size = 7
        self.hidden_size = 50
        self.num_layers = 1
        self.output_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 1000
        self.learning_rate = 0.0015

# 训练模块
class TrainingModule:
    def __init__(self, model, train_dataloader, test_dataloader, criterion, optimizer, params):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.params = params
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.2)  # 学习率调整

    def train_and_evaluate(self):
        best_model = None
        min_test_loss = float('inf')

        train_loss_all = []
        test_loss_all = []
        for epoch in range(self.params.num_epochs):
            self.model.train()
            train_loss = []
            test_loss = []
            for x_batch, y_batch in self.train_dataloader:
                x_batch, y_batch = x_batch.to(self.params.device), y_batch.to(self.params.device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                train_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                for x_batch, y_batch in self.test_dataloader:
                    x_batch, y_batch = x_batch.to(self.params.device), y_batch.to(self.params.device)
                    with torch.no_grad():
                        output = self.model(x_batch)
                        loss = self.criterion(output, y_batch)
                        test_loss.append(loss.item())
                if test_loss[-1] < min_test_loss:
                    min_test_loss = test_loss[-1]
                    best_model = copy.deepcopy(self.model)

            train_loss_all.append(np.mean(train_loss))
            test_loss_all.append(np.mean(test_loss))

            if (epoch+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.params.num_epochs}], '
                      f'Train Loss: {train_loss_all[-1]:.4f}, Test Loss: {test_loss_all[-1]:.4f}')
            self.scheduler.step()  # 学习率调整

        return best_model, train_loss_all, test_loss_all, min_test_loss

# 主函数
def train_main(station=None):
    parser = argparse.ArgumentParser(description='Training script for a PyTorch model.')
    parser.add_argument('--station', type=str, default='s1', help='Station to train the model on')
    args = parser.parse_args()

    params = Parameters()
    if station == None:
        station = args.station
    data = load_data(station)
    print(f'\n**************************** station: {station} ****************************\n')

    train_size = int(0.8 * len(data))
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    # test_size = int(0.8 * len(data))
    # test, train = data.iloc[:test_size], data.iloc[test_size:]

    train_set = TimeSeriesPredictionDataset(train, 24, params.output_size)
    test_set = TimeSeriesPredictionDataset(test, 24, params.output_size)
    train_dataloader = DataLoader(dataset=train_set, batch_size=params.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=params.batch_size, shuffle=True, drop_last=True)


    model = SimpleLSTM(params.input_size, params.hidden_size, params.num_layers, params.output_size).to(params.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    best_model, train_loss, test_loss, min_test_loss = TrainingModule(model, train_dataloader, test_dataloader, criterion, optimizer, params).train_and_evaluate()

    torch.save(best_model, f'./model/model_{station}_{min_test_loss}.pth')

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_loss) + 1), test_loss, range(1, len(test_loss) + 1), np.full(params.num_epochs, 0.01), label='Test Loss')
    plt.title('Test Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./log/model_{station}.png')  # 指定保存路径和文件名

    # plt.show()



def main():
    for station in Stations:
        train_main(station)
    # train_main('s1')



if __name__ == "__main__":
    main()