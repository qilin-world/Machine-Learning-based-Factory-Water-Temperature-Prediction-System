import torch
import torch.nn as nn


# 模型类
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 确保h0和c0在与x相同的设备上
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


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

