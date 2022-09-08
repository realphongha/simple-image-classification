import torch
from torch import nn
from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head


class Model(nn.Module):
    def __init__(self, cfg, training=True):
        super(Model, self).__init__()
        self.cfg = cfg
        self.nc = cfg["DATASET"]["NUM_CLS"]
        self.backbone, self.out_channels = build_backbone(cfg, training, self)
        self.neck, self.out_channels = build_neck(cfg, training, self)
        self.head = build_head(cfg, training, self)

    def freeze(self, parts):
        for part in parts:
            if part not in ("backbone", "neck", "head"):
                raise NotImplementedError("Cannot freeze %s!" % part)
            print("Freezing %s..." % part)
            for name, p in self.named_parameters():
                if part in name:
                    print("Freezing %s..." % name)
                    p.requires_grad = False

    def remove_fc(self):
        if self.cfg["MODEL"]["HEAD"]["NAME"] == "LinearCls":
            self.head.fc = nn.Sequential()
        else:
            raise NotImplementedError("Removing FC for head %s is not implemented!" %
                self.head_name)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    cfg = {
        "MODEL": {
            "BACKBONE": {"NAME": "mobileone", "WIDEN_FACTOR": "s0"},
            "NECK": "GlobalAveragePooling",
            "HEAD": {"NAME": "LinearCls", "DROPOUT": 0.0}
        },
        "DATASET": {"NUM_CLS": 2},
        "TRAIN": {"PRETRAINED": None}
    }
    model = Model(cfg)
    model.remove_fc()

    print(model)
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs)
    print(test_outputs.size())
