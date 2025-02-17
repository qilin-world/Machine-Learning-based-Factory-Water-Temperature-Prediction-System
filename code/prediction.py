import pandas as pd
import torch
import copy
from data_preprocessing import load_data_test, TimeSeriesPredictionDataset
from model import SimpleLSTM, TransformerModel
from train_ql import Parameters
import glob






# 载入模型
Station = ['s1', 's4', 's5', 's8', 's10', 's12', 's14', 's15']
device = 'cpu'
params = Parameters()
model = SimpleLSTM(params.input_size, params.hidden_size, params.num_layers, params.output_size).to(device)




out = pd.DataFrame()
for station in Station:
    # model_directory = './model'
    model_directory = './model/7in_1out_100_1/0.0077'

    model_pattern = f'model_{station}_*.pth'
    model_files = glob.glob(f'{model_directory}/{model_pattern}')
    # print(f'model: {model_files[0]}')
    model = torch.load(model_files[0])
    model.to(device)
    # 设置模型为评估模式
    model.eval()

    # 获取测试数据
    data = load_data_test(station)
    # print(data)


    # 开始推理
    output = []
    batch = TimeSeriesPredictionDataset(data, 24, params.output_size)
    for i, j in zip(data.index[24:], range(len(batch))):
        d = data[j:j + 24]
        d = torch.tensor(d.values, dtype=torch.float32)
        d = d.unsqueeze(0)
        d.to(device)
        if params.output_size == 1:
            o = model(d)
            o = o.item()
            data.loc[i, '瞬时热量'] = o
        else:
            o = model(d)
            o = o[0,:]
            o = o.tolist()
            data.loc[i, '一网供水温度'] = o[0]
            data.loc[i, '瞬时热量'] = o[1]
        # print(data)

    # 整理数据
    data = data.tail(24)
    out_1 = pd.DataFrame()
    out_1['换热站ID'] = station
    out_1['时间'] = data.index
    out_1['换热站ID'] = station
    for i, j in zip(out_1.index, data.index):
        out_1.loc[i, '瞬时热量'] = data.loc[j, '瞬时热量']
    #print(out_1)
    out=pd.concat([out, out_1])
    #print(out)


'''
保存文件
'''
out = out.sort_values(by=['换热站ID', '时间'])
out.pop('换热站ID')
out.pop('时间')
# out['瞬时热量'] = out['瞬时热量'] + 2  # 数据反映射

out.index = range(192)
print(out)

# 总文件
out.to_csv('output/result.csv', header=False)

# 分别文件
num_files = (len(out) + 23) // 24  # 计算需要保存的文件数量
for i in range(num_files):
    start_idx = i * 24
    end_idx = start_idx + 24
    # 切片DataFrame来获取每24行数据
    slice_out = out.iloc[start_idx:end_idx]
    # 构造文件名
    filename = f'output/some/result_{i+1}.csv'
    # 保存到CSV，不包含header
    slice_out.to_csv(filename, header=False)