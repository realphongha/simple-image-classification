import torch
from torch import nn


class BCnnNeck(nn.Module):
    # Bilinear CNNs for Fine-grained Visual Recognition https://arxiv.org/pdf/1504.07889v6.pdf
    # Implemented by HaoMood https://github.com/HaoMood/bilinear-cnn
    def __init__(self):
        super(BCnnNeck, self).__init__()

    def forward(self, x):
        N, c, h, w = x.size()
        x = x.view(N, c, h*w)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (h*w)  # Bi-linear CNN
        x = x.view(N, c ** 2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        return x
