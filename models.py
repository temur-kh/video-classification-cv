import torch
import torch.nn as nn
import torchvision.models as models


class AvgCNNModel(nn.Module):
    def __init__(self, output_size=101):
        super(self.__class__, self).__init__()
        self.output_size = output_size
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size, bias=True)

    def forward(self, videos):
        output = []
        for video in videos:
            out = self.cnn(video).view(-1, self.output_size)
            output.append(torch.mean(out, dim=0))
        return torch.stack(output)


class CRNNModel:
    pass
