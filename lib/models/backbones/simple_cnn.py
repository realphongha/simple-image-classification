import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import load_checkpoint


class SimpleCNN(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        return x


def get_simple_cnn_backbone(model_size, img_channels, pretrained=None):
    if model_size == "1.0x":
        model = SimpleCNN(img_channels)
        out_channels = 64
    else:
        raise NotImplementedError(f"Model size {model_size} is not supported!")
    if pretrained:
        load_checkpoint(model, pretrained, strict=False)
    return model, out_channels


if __name__ == "__main__":
    model, out_channels = get_simple_cnn_backbone("1.0x", 1)
    print(model)
    print(out_channels)
    print("Params:", sum(p.numel() for p in model.parameters()))

    test_data = torch.rand(5, 1, 28, 28)
    test_outputs = model(test_data)
    print(test_outputs.size())

