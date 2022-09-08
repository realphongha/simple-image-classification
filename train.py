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
import torch.optim as optim

from torch.utils.data import DataLoader

from lib.datasets import DATASETS
from lib.models.model import Model
from lib.losses import build_loss
from lib.tools import train, evaluate
from lib.utils import save_checkpoint
from lib.scheduler import build_scheduler


def main(cfg):
    cudnn.benchmark = cfg["CUDNN"]["BENCHMARK"]
    cudnn.deterministic = cfg["CUDNN"]["DETERMINISTIC"]
    cudnn.enabled = cfg["CUDNN"]["ENABLED"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if cfg["GPUS"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUS"]

    device = 'cuda' if (torch.cuda.is_available() and cfg["GPUS"]) else 'cpu'
    print("Start training using device: %s" % device)
    print("Config:", cfg)

    if cfg["DATASET"]["NAME"] in DATASETS:
        Ds = DATASETS[cfg["DATASET"]["NAME"]]
        train_ds = Ds(data_path=cfg["DATASET"]["TRAIN"],
                      is_train=True,
                      cfg=cfg)
        val_ds = Ds(data_path=cfg["DATASET"]["VAL"],
                    is_train=False,
                    cfg=cfg)
    else:
        raise NotImplementedError("%s is not implemented!" %
                                  cfg["DATASET"]["NAME"])

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["TRAIN"]["BATCH_SIZE"],
                              shuffle=cfg["TRAIN"]["SHUFFLE"],
                              num_workers=cfg["WORKERS"])
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["TEST"]["BATCH_SIZE"],
                            shuffle=cfg["TEST"]["SHUFFLE"],
                            num_workers=cfg["WORKERS"])

    model = Model(cfg, training=True)
    model.to(device)

    if cfg["TRAIN"]["OPTIMIZER"] == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg["TRAIN"]["LR"])
    elif cfg["TRAIN"]["OPTIMIZER"] == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg["TRAIN"]["LR"],
                              momentum=cfg["TRAIN"]["MOMENTUM"],
                              weight_decay=cfg["TRAIN"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError("%s is not implemented!" %
                                  cfg["TRAIN"]["OPTIMIZER"])

    criterion = build_loss(cfg, device)

    metric = cfg["TEST"]["METRIC"]
    begin_epoch = 0
    last_epoch = -1
    best_acc = -1
    best_clf_report = None
    best_conf_matrix = None
    train_loss = list()
    val_loss = list()
    train_acc = list()
    val_acc = list()

    # Tries to resume or creates new output directory
    if cfg["TRAIN"]["AUTO_RESUME"] and cfg["TRAIN"]["CKPT"]:
        ckpt_file = cfg["TRAIN"]["CKPT"]
        if not ckpt_file.endswith(".pth"):
            ckpt_file = os.path.join(ckpt_file, "last.pth")
        output_path = os.path.split(ckpt_file)[0]
        print("=> loading checkpoint from %s..." % ckpt_file)
        checkpoint = torch.load(ckpt_file, map_location=device)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        best_acc, best_clf_report, best_conf_matrix, train_loss, val_loss, train_acc, val_acc = best_perf
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            ckpt_file, checkpoint['epoch']))
    else:
        datetime_str = datetime.datetime.now().strftime("--%Y-%m-%d--%H-%M")
        output_path = os.path.join(os.path.join(cfg["OUTPUT"], "train"),
                                cfg["DATASET"]["NAME"] + "--" +
                                cfg["MODEL"]["BACKBONE"]["NAME"] + "--" +
                                cfg["MODEL"]["NECK"] + "--" +
                                cfg["MODEL"]["HEAD"]["NAME"] + "--" +
                                datetime_str)
        os.makedirs(output_path, exist_ok=False)
        with open(os.path.join(output_path, "configs.txt"), "w") as output_file:
            json.dump(cfg, output_file)

    lr_scheduler = build_scheduler(cfg, optimizer, last_epoch)

    if cfg["MODEL"]["FREEZE"]:
        parts_to_freeze = cfg["MODEL"]["FREEZE"].strip().split(",")
        parts_to_freeze = [p.strip() for p in parts_to_freeze]
        model.freeze(parts_to_freeze)

    warmup_freeze_eps = cfg["TRAIN"]["WARMUP_FREEZE"]["EPOCHS"]
    warmup_freeze_parts = cfg["TRAIN"]["WARMUP_FREEZE"]["PARTS"]
    if warmup_freeze_eps and warmup_freeze_parts:
        warmup_freeze = True
        warmup_freeze_parts = warmup_freeze_parts.strip().split(",")
        warmup_freeze_parts = [p.strip() for p in warmup_freeze_parts]
    else:
        warmup_freeze = False
    frozen = False

    model.to(device)

    for epoch in range(begin_epoch, cfg["TRAIN"]["EPOCHS"]):
        print("EPOCH %i:" % epoch)

        # freezes model weights for warmup
        if warmup_freeze:
            if epoch < warmup_freeze_eps and not frozen:
                model.freeze(warmup_freeze_parts)
                frozen = True
            elif epoch >= warmup_freeze_eps and frozen:
                model.free(warmup_freeze_parts)
                frozen = False

        # trains
        f1, acc, loss, conf_matrix = train(model, criterion, optimizer, train_loader, device)
        if metric == "acc":
            m = acc
        elif metric == "f1":
            m = f1
        else:
            raise NotImplementedError("Metric %s is not implemented!" % metric)
        train_acc.append(m)
        train_loss.append(loss)
        lr_scheduler.step()

        # evaluates
        f1, acc, clf_report, loss, conf_matrix = evaluate(model, criterion,
                                                          val_loader, device)
        if metric == "acc":
            m = acc
        elif metric == "f1":
            m = f1
        else:
            raise NotImplementedError("Metric %s is not implemented!" % metric)
        val_acc.append(m)
        val_loss.append(loss)
        best_model = False
        if m > best_acc:
            best_acc = m
            best_clf_report = clf_report
            best_conf_matrix = conf_matrix
            best_model = True

        # saves checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'perf': (best_acc, best_clf_report, best_conf_matrix, train_loss, val_loss, train_acc, val_acc),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, best_model, output_path, cfg["SAVE_ALL_EPOCHES"])

        print()

        # writes results:
        with open(os.path.join(output_path, "final_results.txt"), "w") as file:
            file.write("%s: %f\n\n" % (metric, best_acc))
            file.write(best_clf_report)
            file.write("\n")
            file.close()

        epochs = range(epoch+1)

        fig = plt.figure()
        plt.plot(epochs, train_acc, 'r', label='Training %s' % metric)
        plt.plot(epochs, val_acc, 'b', label='Validation %s' % metric)
        plt.title('Training and validation %s' % metric)
        plt.legend()
        fig.savefig(os.path.join(output_path, 'acc_plot.png'),
                    bbox_inches='tight')

        fig = plt.figure()
        plt.plot(epochs, train_loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        fig.savefig(os.path.join(output_path, 'loss_plot.png'),
                    bbox_inches='tight')

        fig = plt.figure()
        df_cm = pd.DataFrame(best_conf_matrix, range(best_conf_matrix.shape[0]),
                            range(best_conf_matrix.shape[0]))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='.3f') # font size
        fig.savefig(os.path.join(output_path, 'confusion_matrix.png'),
                    bbox_inches='tight')

        plt.close("all")

        print()

    print("Done training!")
    print("Best %s:" % metric, best_acc)
    print(best_clf_report)


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

    main(cfg)
