import torch
from torch.nn import CrossEntropyLoss
from .focal_loss import FocalLoss


def build_loss(cfg, device):
    if cfg["MODEL"]["HEAD"]["LOSS"] == "CrossEntropy":
        loss_weight = cfg["MODEL"]["HEAD"]["LOSS_WEIGHT"]
        if loss_weight:
            criterion = CrossEntropyLoss(weight=torch.Tensor(loss_weight).to(device))
        else:
            criterion = CrossEntropyLoss()
    elif cfg["MODEL"]["HEAD"]["LOSS"] == "FocalLoss":
        gamma = cfg["MODEL"]["HEAD"]["LOSS_GAMMA"]
        alpha = cfg["MODEL"]["HEAD"]["LOSS_ALPHA"]
        if not gamma:
            gamma = 2
        if not alpha:
            alpha = None
        criterion = FocalLoss(gamma=gamma, alpha=torch.Tensor(alpha).to(device))
    else:
        raise NotImplementedError("%s is not implemented!" % cfg["MODEL"]["HEAD"]["LOSS"])
    return criterion
