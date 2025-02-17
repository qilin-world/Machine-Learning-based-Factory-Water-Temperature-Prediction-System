import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


# 加载数据
Stations = {'s1', 's4', 's5', 's8', 's10', 's12', 's14', 's15'}
station_columns = ['瞬时热量']
weather_columns = ['气温', '风力等级', '日出时间', '日落时间', '天气', '小时']

# station_columns = ['一网供水温度', '一网回水温度', '一网供水压力', '一网回水压力', '瞬时流量',  '瞬时热量']
# station_columns = ['一网供水温度', '瞬时热量']
# station_columns = ['一网回水温度']
# weather_columns = ['气温', '风力等级', '日出时间', '日落时间', '天气']

ultimate_Order = weather_columns + station_columns


# 天气映射函数
def map_weather_condition(weather_condition):
    weather_map = {
        '晴': 1.0,
        '多云': 0.5,
    }
    return weather_map.get(weather_condition, 0)

# 处理日出和日落时间，转换为分钟
def convert_to_minutes(t):
    if pd.isna(t):
        return None
    return (t.hour * 60 + t.minute)

# 数据归一化
def normalize_data_ds(data, min, max):
    data = (data - min) / (max - min) * 16.0
    return data
def normalize_data_dw(data, min, max):
    data = (data - min) / (max - min)
    return data

# 训练集
def load_data(station):
    if station not in Stations:
        return None
    '''
    读取站点数据
    '''
    ds = pd.read_csv('./data/dataset/station.csv')
    ds = ds[ds['换热站ID'] == station]
    ds.index = pd.to_datetime(ds.pop('时间'), format='%Y-%m-%d %H:%M:%S')

    # # 数据归一化
    # min_max_values = {
    #     '一网供水温度': (60, 100),
    #     '一网回水温度': (30, 50),
    #     '一网供水压力': (0.4, 0.6),
    #     '一网回水压力': (0.3, 0.6),
    #     '瞬时流量': (16, 100),
    #     '瞬时热量': (2, 18)
    # }
    # for column in station_columns:
    #     min_val, max_val = min_max_values[column]
    #     ds[column] = normalize_data_ds(ds[column], min_val, max_val)


    ds = ds[station_columns]
    # ds = ds.resample('h').mean()
    ds = ds[::60]


    '''
    读取天气数据
    '''
    dw = pd.read_csv('./data/dataset/weather.csv')
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    dw['天气'] = dw['天气'].apply(map_weather_condition)
    dw = dw.dropna(subset=['天气'])  # 删除没有映射的天气数据

    dw['小时'] = dw.index.hour
    dw['分钟'] = dw.index.minute
    dw['日出时间'] = dw['日出时间'].apply(lambda x: convert_to_minutes(pd.to_datetime(x)))
    dw['日落时间'] = dw['日落时间'].apply(lambda x: convert_to_minutes(pd.to_datetime(x)))

    # 数据归一化
    dw['气温'] = normalize_data_dw(dw['气温'], -21, 3)
    dw['风力等级'] = normalize_data_dw(dw['风力等级'], 0, 3)
    dw['日出时间'] = normalize_data_dw(dw['日出时间'], 435, 445)
    dw['日落时间'] = normalize_data_dw(dw['日落时间'], 1014, 1023)
    dw['小时'] = normalize_data_dw(dw['小时'], 0, 23)

    dw = dw[weather_columns]
    data = pd.concat([dw, ds], axis=1)
    data = data[ultimate_Order]
    # print(data)
    # data.interpolate(method='linear', inplace=True)
    return data

# 测试集
def load_data_test(station):
    if station not in Stations:
        return None

    # 读取天气数据
    dw = pd.read_csv('./data/datatest/weather.csv')
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    dw['天气'] = dw['天气'].apply(map_weather_condition)
    dw = dw.dropna(subset=['天气'])  # 删除没有映射的天气数据

    dw['小时'] = dw.index.hour
    dw['分钟'] = dw.index.minute
    dw['日出时间'] = dw['日出时间'].apply(lambda x: convert_to_minutes(pd.to_datetime(x)))
    dw['日落时间'] = dw['日落时间'].apply(lambda x: convert_to_minutes(pd.to_datetime(x)))

    # 数据归一化
    dw['气温'] = normalize_data_dw(dw['气温'], -21, 3)
    dw['风力等级'] = normalize_data_dw(dw['风力等级'], 0, 3)
    dw['日出时间'] = normalize_data_dw(dw['日出时间'], 435, 445)
    dw['日落时间'] = normalize_data_dw(dw['日落时间'], 1014, 1023)
    dw['小时'] = normalize_data_dw(dw['小时'], 0, 23)

    dw = dw[weather_columns]

    # 过去24小时数据与当前24小时拼起来
    data = load_data(station)

    # 确保两个 DataFrame 具有相同的列
    if not data.columns.equals(dw.columns):
        # print("Columns are not aligned:", data.columns, dw.columns)
        # 这里可以添加代码来调整列，例如添加缺失的列或删除多余的列
        # 例如，如果 dw 缺少某些列，可以添加这些列并填充默认值
        for col in data.columns:
            if col not in dw.columns:
                dw[col] = 0  # 或者其他合适的默认值
    # print(dw)
    # print(f'data: {data}')
    data = pd.concat([data.tail(24), dw], sort=False)
    data = data[ultimate_Order]
    # data.interpolate(method='linear', inplace=True)
    # print(data)

    return data


# 数据集类
class TimeSeriesPredictionDataset(Dataset):
    def __init__(self, data, seq_length, output_size):
        self.data = data.values
        self.seq_length = seq_length
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        if self.output_size == 1:
            y = [self.data[idx + self.seq_length][-1]]
        else:
            y = self.data[idx + self.seq_length][-self.output_size:]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)