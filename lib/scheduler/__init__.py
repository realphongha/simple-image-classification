import torch.optim as optim
from .lr_scheduler import *


def build_scheduler(cfg, optimizer, last_epoch):
    if cfg["TRAIN"]["LR_SCHEDULE"] == "multistep":
        if cfg["TRAIN"]["WARMUP"]:
            lr_scheduler = WarmupMultiStepSchedule(
                optimizer, cfg["TRAIN"]["WARMUP"], cfg["TRAIN"]["LR_STEP"], 
                cfg["TRAIN"]["LR_FACTOR"],
                last_epoch=last_epoch
            )
        else:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, cfg["TRAIN"]["LR_STEP"], cfg["TRAIN"]["LR_FACTOR"],
                last_epoch=last_epoch
            )
    else:
        raise NotImplementedError("%s lr scheduler is not implemented!" % \
            cfg["TRAIN"]["LR_SCHEDULE"])
    return lr_scheduler
