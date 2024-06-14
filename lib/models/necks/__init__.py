import torch.nn as nn
from .b_cnn import BCnnNeck


def build_neck(cfg, training, model):
    neck_name = cfg["MODEL"]["NECK"]
    out_channels = model.out_channels
    if neck_name == "GlobalAveragePooling":
        neck = nn.AvgPool2d(7)
    elif not neck_name or neck_name == "no":
        cfg["MODEL"]["NECK"] = "none"
        neck = None
    elif neck_name == "B-CNN":
        neck = BCnnNeck()
        out_channels **= 2
    else:
        raise NotImplementedError("Neck %s is not implemented!" % neck_name)
    return neck, out_channels
