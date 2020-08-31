import torch
import torch.nn as nn
import torchvision.models as models


class AvgCNNModel(nn.Module):
    def __init__(self, output_size=101, frames_cnt=32):
        super(self.__class__, self).__init__()
        self.output_size = output_size
        self.frames_cnt = frames_cnt
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size, bias=True)

    def forward(self, videos):
        out = self.cnn(videos)
        out = torch.mean(out.reshape(-1, self.frames_cnt, self.output_size), dim=1)
        return out


class CNNtoRNNModel(nn.Module):
    def __init__(self, output_size=101, frames_cnt=32, drop_rate=0.3, rnn_hid_size=128, rnn_num_layers=2):
        super(self.__class__, self).__init__()
        self.frames_cnt = frames_cnt

        self.cnn = models.resnet18(pretrained=True)
        cnn_out_size = self.cnn.fc.in_features
        self.cnn.fc = Identity()

        self.rnn = nn.LSTM(cnn_out_size, rnn_hid_size, rnn_num_layers)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(rnn_hid_size, output_size)

    def forward(self, videos):
        b_z, c, h, w = videos.shape
        x = videos.reshape(-1, self.frames_cnt, c, h, w)
        ii = 0
        y = self.cnn((x[:, ii]))
        out, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, self.frames_cnt):
            y = self.cnn((x[:, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, x):
        return x
