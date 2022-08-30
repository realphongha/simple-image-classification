import os
import datetime
import json

import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from lib.datasets import DATASETS
from lib.models.model import Model
from lib.losses import LOSSES
from lib.tools import evaluate


def main(cfg, output_path):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if cfg["GPUS"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]

    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start evaluating using device: %s" % device)
    print("Config:", cfg)

    if cfg["DATASET"]["NAME"] in DATASETS:
        Ds = DATASETS[cfg["DATASET"]["NAME"]]
        val_ds = Ds(data_path=cfg["DATASET"]["VAL"],
                    is_train=False,
                    cfg=cfg)
    else:
        raise NotImplementedError("%s is not implemented!" %
                                  cfg["DATASET"]["NAME"])

    val_loader = DataLoader(val_ds,
                            batch_size=cfg["TEST"]["BATCH_SIZE"],
                            shuffle=cfg["TEST"]["SHUFFLE"],
                            num_workers=cfg["WORKERS"])

    model = Model(cfg, training=False)

    weights_path = cfg["TEST"]["WEIGHTS"]
    if not weights_path:
        raise Exception("Please specify path to model weights in config file!")
    weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights['state_dict'] if 'state_dict' in weights else weights)

    model.to(device)

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
        criterion = LOSSES["FocalLoss"](gamma=gamma, alpha=torch.Tensor(alpha).to(device))
    else:
        raise NotImplementedError("%s is not implemented!" % cfg["MODEL"]["HEAD"]["LOSS"])

    # evaluates
    f1, acc, clf_report, loss, conf_matrix = evaluate(model, criterion,
                                                        val_loader, device)

    print("Done evaluating!")

    # writes results:
    with open(os.path.join(output_path, "final_results.txt"), "w") as file:
        file.write("Acc: %f\n\n" % acc)
        file.write(clf_report)
        file.write("\n")
        file.close()

    fig = plt.figure()
    df_cm = pd.DataFrame(conf_matrix, range(conf_matrix.shape[0]),
                        range(conf_matrix.shape[0]))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='.3f') # font size
    fig.savefig(os.path.join(output_path, 'confusion_matrix.png'),
                bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml',
                        help='path to config file')
    opt = parser.parse_args()

    with open(opt.config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
    output_path = os.path.join(os.path.join(cfg["OUTPUT"], "val"),
                               cfg["DATASET"]["NAME"] + "--" +
                               cfg["MODEL"]["BACKBONE"]["NAME"] + "--" +
                               cfg["MODEL"]["NECK"] + "--" +
                               cfg["MODEL"]["HEAD"]["NAME"] + "--" +
                               datetime_str)
    os.makedirs(output_path, exist_ok=False)
    with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
        json.dump(cfg, output_file)

    main(cfg, output_path)
