from torch import nn


class LinearClsHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super(LinearClsHead, self).__init__()
        self.dropout_or_not = dropout > 0.0
        self.in_channels = in_channels
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if self.dropout_or_not:
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.in_channels)
        x = self.fc(x)
        return x
