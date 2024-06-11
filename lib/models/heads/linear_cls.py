from torch import nn


class LinearClsHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super(LinearClsHead, self).__init__()
        self.dropout_or_not = dropout > 0.0
        self.in_channels = in_channels
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

        # self.fc1 = nn.Linear(in_channels, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.dropout_or_not:
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.in_channels)
        x = self.fc(x)
        return x

    # def forward(self, x):
    #     if self.dropout_or_not:
    #         x = self.dropout(x)
    #     x = x.contiguous().view(-1, self.in_channels)
    #     x = self.fc1(x)
    #     x = self.bn1(x)
    #     x = self.relu1(x)
    #     x = self.fc2(x)
    #     return x

