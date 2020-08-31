import torch
import torch.nn as nn
import torchvision.models as models


class AvgCNNModel(nn.Module):
    def __init__(self, output_size=101, batch_size=16, frames_cnt=32):
        super(self.__class__, self).__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.frames_cnt = frames_cnt
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size, bias=True)

    def forward(self, videos):
        out = self.cnn(videos).view(-1, self.output_size)
        out = torch.mean(out.reshape(-1, self.frames_cnt, self.output_size), dim=1)
        return out


class CRNNModel:
    pass
