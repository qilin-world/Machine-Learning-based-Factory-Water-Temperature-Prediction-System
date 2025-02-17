# 一种基于深度学习的供暖热量预测方法


# 🔥 参考资料

我们在写代码时参考了一些资料，在此列出:
  - [Transformer 快速入门](https://transformers.run/)
  - [Transformer官方代码](https://github.com/huggingface/transformers)
 

# 🧐 环境设置

为了运行我们的程序，请保证您的电脑装有Anaconda，建议使用pycharm软件打开，同时在终端中运行下面指令以获得和我们相同的环境：

```
conda create -n prediction_ql python=3.6
conda activate prediction_ql
pip install -r requirments.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# 数据

数据来源于山东大学人工智能与机器人实验室，训练数据地址：

```
./data/dataset/station.csv
./data/dataset/weather.csv
```

测试数据地址：

```
./data/datatest/weather.csv
```

## 预训练模型

我们在程序中保留了一些我们预训练模型，部署他们，您可以直接展开测试，模型文件地址：

```
./model
```

## 🎯 模型训练

运行下面指令，可以快速开展模型训练：

```
python .\code\train_ql.py
```

训练过程中的损失函数图像储存在：

```
./log
```

如果您想调整超参数，下面提供几种指南，帮助您快速修改

1.调节模型参数，只需在'.\code\train_ql.py'文件中修改如下参数

```
class Parameters:
    def __init__(self):
        self.batch_size = 50       # 训练批次（如果电脑运行不动，请将该参数调小）
        self.input_size = 7        # 输入数据维度
        self.hidden_size = 100     # 隐藏层参数
        self.num_layers = 2        # 隐藏层参数（层数）
        self.output_size = 1       # 输出数据维度
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断电脑是否有gpu，自动帮您调整
        self.num_epochs = 1000     # 总的训练轮数
        self.learning_rate = 0.02  # 初始学习率
```

2.选择不同输入数据，您只需先将1.所示数据输入维度调整至您想要的大小，然后在'.\code\data_preprocessing.py'文件中调整输入数据

```
Stations = {'s1', 's4', 's5', 's8', 's10', 's12', 's14', 's15'}             # 要训练的管道
station_columns = ['瞬时热量']                                               # 输出数据
weather_columns = ['气温', '风力等级', '日出时间', '日落时间', '天气', '小时']  # 输入的weather数据
ultimate_Order = weather_columns + station_columns                           # 完整输入数据

```

3.选择不同输出数据，参照2.方式，您可以额外加入针对station数据的归一化（映射到[0,16]），

另外需要在'.\code\data_preprocessing.py'中将下列部分解注释

```
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
```

还需在'.\code\prediction.py'文件中将热量数据反映射注释解开

```
# out['瞬时热量'] = out['瞬时热量'] + 2  # 数据反映射
```

4.选择其他model进行训练，只需参照'.\code\model.py'文件中

```
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=24 * 3, nhead=8, num_encoder_layers=2, dim_feedforward=240):
        super(TransformerModel, self).__init__()
        # self.input_linear = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # 调整输入形状以匹配 Transformer 的期望输入
        # print(f'x.size: {x.shape}')
        # x = self.input_linear(x).unsqueeze(1)  # [batch_size, seq_len=1, d_model]
        # x = x.transpose(0, 1)  # [seq_len=1, batch_size, d_model]

        x = x.reshape(x.shape[0], -1)
        x = x.unsqueeze(0)
        x = self.transformer(x)  # [seq_len=1, batch_size, d_model]
        x = x.mean(dim=0)  # [batch_size, d_model]
        x = self.fc_out(x)  # [batch_size, 1]
        return x
```

5.另外，我们还对原始的输入数据全部做了处理，提高了气温等数据精度，具体处理方式采取了滤波与拟合的办法，

如果您想使用，请参考文件

```
./data/dataset/weather_dataset.csv
./data/datatest/weather_datatest.csv
```

处理的目的很显然，原始的气温等数据并不是“连续”的，精度精确到个位数，这可能会影响预测的质量

## 🚨 模型测试

快速开展模型测试，并产生'.\output/result.csv'文件,只需运行下面指令：

```
python .\code\prediction.py
```

## 📝 写在最后

我们尽最大努力将程序模块化并减少使用者上手难度，但由于时间有限和作者能力有限，程序难免可能有bug，如果您遇到问题，可以联系：

邮箱：812997956@qq.com

作者：（山东大学） 王琪霖、彭昊
