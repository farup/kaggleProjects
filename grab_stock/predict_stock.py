import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

import tqdm
import datetime 
import torch 
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Dataset, TensorDataset 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def str_to_datetime(s): 
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe, first_date_str='2022-01-01', last_date_str='2023-11-29', n=4):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    check_ = len(dataframe) % n
    period_intres = dataframe.loc[first_date_str: last_date_str]
    check_ = len(period_intres) % n

    if check_ != 0: 
        period_intres = period_intres[check_:]

    dates = []
    X, Y = [], []
    target_date = first_date
    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)
        values = df_subset['Close']
        x, y = values[:-1], values[-1]

        next_datetime_str = df_subset.index.values[-1]
        next_date_str = np.datetime_as_string(next_datetime_str).split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year)) + datetime.timedelta(days=n + 1)
        dates.append(target_date)
        X.append(x)
        Y.append(y)

        if last_time: 
            break

        target_date = next_date

        if target_date >= last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n -1):
        X[:, i]
        ret_df[f'Target-{n-1-i}'] = X[:, i]
    ret_df['Target'] = Y

    return ret_df, X, Y


def windowed_df_to_date_X_y(windowed_df):
    df_as_np = windowed_df.to_numpy()

    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:,1:-1]
    X = middle_matrix.reshape(len(dates), middle_matrix.shape[1], 1)


    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

def custom_collate(batch):
    data = torch.stack([item['Date'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'data': data, 'label': labels}


def save_model(model,n, output_folder, epoch=None):
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    if epoch:  
        torch.save(model.state_dict(), os.path.join(output_folder, f'Stock_state_dict_epoch_{epoch}.pt'))
        print('Saved!')
    else: 
        torch.save(model.state_dict(), os.path.join(output_folder, f'Stock_state_dict_{n}.pt'))
        print('Saved!')

def load_model(state_dict, input_size): 
    if type(state_dict) == str: 
        state_dict = torch.load(state_dict)
    model = LSTM_Stock(input_size=input_size)
    model.load_state_dict(state_dict)
    model.cuda()
    return model

def infer(model, input, user_input=False):

    if user_input:
        price3 = input('Enter price 2 days ago: ') 
        price2 = input('Enter price 1 days ago: ') 
        price1 = input('Enter price today: ') 

        input = np.array(price3, price2, price1)

    input = torch.from_numpy(input).unsqueeze(0).reshape(1,1,3).to(torch.float32).to(device)
    prediction = model(input)
    return prediction



@torch.no_grad()
def eval(model, data_loader_eval, loss_track_eval, epoch, batch_size, n):
    
    for batch_data, batch_labels in tqdm.tqdm(data_loader_eval):
        batch_data = batch_data.reshape(batch_size,1,n).to(torch.float32).to(device)
        output = model(batch_data)
        loss_output = loss(output, batch_labels.to(torch.float32).to(device))
        loss_track_eval.append(loss_output.cpu().numpy())
    
    #save_model(model, output_folder='./model_eval', epoch=epoch)


def train(model, data_loader_train, data_loader_eval, n, batch_size, epochs=10):
    loss_track_train = []
    loss_track_eval = []
    for epoch in range(epochs): 
        for batch_data, batch_labels in tqdm.tqdm(data_loader_train):
            batch_data = batch_data.reshape(batch_size, 1,n).to(torch.float32).to(device)
            output = model(batch_data)
            loss_output = loss(output, batch_labels.to(torch.float32).to(device))

            loss_track_train.append(loss_output.detach().cpu().numpy())
            
            optimizer.zero_grad() # compute gradients
            loss_output.backward()
            optimizer.step()

        if epoch % 2:
            print(f'====== Eval epoch number {epoch}/{epochs} ======')
            eval(model, data_loader_eval, loss_track_eval, epoch, batch_size, n)

    return model, loss_track_eval, loss_track_train

class LSTM_Stock(nn.Module): 
    def __init__(self, input_size=3, hidden_size=32, output_size=1) -> None:
        super().__init__()


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

        self.float()
        # self.model = nn.Sequential(
        #     nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        #     nn.Linear(hidden_size, output_size)  
        # )

        print('hello')

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.fc1(x)
        return x

class StockDataset(Dataset): 
    def __init__(self, X, Y) -> None:
        super().__init__()
        # self.dataframe = dataframe
        # self.features = dataframe.iloc[:, 1:-1].to_numpy()
        # self.labels = dataframe.iloc[:, -1].to_numpy()
     
        self.features = X
        self.labels = np.asarray(Y)
        print('yo')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.features[index], self.labels[index]
        
if __name__ == "__main__": 

    
    df = pd.read_csv(r'C:\Users\tnf\Documents\KodeProsjekter\2023\Kaggle\grab_stock\GRAB.csv')

    df['Date'] = df['Date'].apply(lambda x: str_to_datetime(x))

    print(f'First date {df.iloc[0]}\n Last date {df.iloc[-1]}')

    df.index = df.pop('Date') # pops are set Date as index column

    n = 10
    batch_size = 1

    ret_df, X , Y = df_to_windowed_df(df, first_date_str='2022-01-01', last_date_str='2023-11-29', n=n)    

    train_ret_df = int(len(ret_df) * .8)
    eval_ret_df = int(len(ret_df) * .9)
    
  
    model = LSTM_Stock(input_size=n)
    loss = nn.MSELoss()

    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

   # dataset_stock_train = StockDataset(ret_df[:train_ret_df])  
    dataset_stock_train = StockDataset(X[:train_ret_df], Y[:train_ret_df])  
    #dataset_stock_test = StockDataset(ret_df[train_ret_df:eval_ret_df])
    dataset_stock_test = StockDataset(X[train_ret_df:eval_ret_df], Y[train_ret_df:eval_ret_df])

    data_loader_train = DataLoader(dataset_stock_test, batch_size=batch_size)
    data_loader_eval = DataLoader(dataset_stock_test, batch_size=batch_size)

    model, loss_train, loss_eval = train(model, data_loader_train, data_loader_eval, n , batch_size,)

    save_model(model, n, output_folder='./model_final')
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(loss_eval, label='Eval', color='Blue')
    ax[0].set_title('Evaluation')
    ax[0].legend()

    ax[1].plot(loss_train, label='Train', color='Red')
    ax[1].set_title('Train')
    ax[1].legend()

    plt.tight_layout()
    plt.show()




