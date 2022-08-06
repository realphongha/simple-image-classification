import torch
from torch import nn
from .backbones import *
from .necks import *
from .heads import *


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        backbone_name = cfg["MODEL"]["BACKBONE"]["NAME"]
        self.neck_name = cfg["MODEL"]["NECK"]
        self.head_name = cfg["MODEL"]["HEAD"]["NAME"]
        self.nc = cfg["DATASET"]["NUM_CLS"]
        if backbone_name == "shufflenetv2":
            self.backbone = get_shufflenetv2(
                cfg["MODEL"]["BACKBONE"]["WIDEN_FACTOR"], 
                cfg["TRAIN"]["PRETRAINED"]
            )
            self.out_channels = self.backbone.stage_out_channels[-1]
        else:
            raise NotImplementedError("Backbone %s is not implemented!" % 
                backbone_name)
        if self.neck_name == "GlobalAveragePooling":
            self.neck = nn.AvgPool2d(7)
        elif self.neck_name == "B-CNN":
            self.neck = BCnnNeck()
            self.out_channels **= 2
        else:
            raise NotImplementedError("Neck %s is not implemented!" % 
                self.neck_name)
        if self.head_name == "LinearCls":
            self.head = LinearClsHead(self.out_channels, self.nc, 
                cfg["MODEL"]["HEAD"]["DROPOUT"])
        else:
            raise NotImplementedError("Head %s is not implemented!" % 
                self.head_name)        

    def freeze(self, parts):
        for part in parts:
            if part not in ("backbone", "neck", "head"):
                raise NotImplementedError("Cannot freeze %s!" % part)
            for name, p in model.named_parameters():
                if part in name:
                    p.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    cfg = {
        "MODEL": {
            "BACKBONE": {"NAME": "shufflenetv2", "WIDEN_FACTOR": "1.0x"},
            "NECK": "B-CNN",
            "HEAD": {"NAME": "LinearCls", "DROPOUT": 0.0}
        }, 
        "DATASET": {"NUM_CLS": 2},
        "TRAIN": {"PRETRAINED": None}
    }
    model = Model(cfg)

    print(model)
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs)
    print(test_outputs.size())
        