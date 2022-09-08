import torch.nn as nn
from .linear_cls import LinearClsHead


def build_head(cfg, training, model):
    head_name = cfg["MODEL"]["HEAD"]["NAME"]
    if head_name == "LinearCls":
        head = LinearClsHead(model.out_channels, model.nc,
            cfg["MODEL"]["HEAD"]["DROPOUT"])
    else:
        raise NotImplementedError("Head %s is not implemented!" % head_name)
    return head
    