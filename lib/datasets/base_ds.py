import random
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from ..transforms import *


class BaseDs(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, cfg):
        self.data_path = data_path
        self.is_train = is_train
        self.cfg = cfg
        self.gray = cfg["DATASET"].get("GRAY", False)
        self.cls = cfg["DATASET"]["CLS"].strip().split(",")
        self.cls = [c.strip() for c in self.cls]
        self.cls_dict = dict()
        for i, cls in enumerate(self.cls):
            self.cls_dict[cls] = i
        self.input_shape = self.cfg["MODEL"]["INPUT_SHAPE"]
        self.data = list()
        self.labels = list()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        fp = self.data[index]
        label = self.labels[index]
        raw_img = cv2.imread(fp)
        pipeline = list()
        safe_pipeline = list()  # safe transform pipeline for augmentation exception
        if self.cfg["DATASET"]["FLIP"]:
            pipeline.append(transforms.RandomHorizontalFlip(0.5))
            safe_pipeline.append(transforms.RandomHorizontalFlip(0.5))
        if self.is_train:
            if self.cfg["DATASET"]["GRAYSCALE"] and self.cfg["DATASET"]["GRAYSCALE"] > random.random():
                pipeline.append(transforms.Grayscale())
            if self.cfg["DATASET"]["COLORJITTER"] and self.cfg["DATASET"]["COLORJITTER"]["PROB"] > random.random():
                pipeline.append(transforms.ColorJitter(
                    brightness=self.cfg["DATASET"]["COLORJITTER"]["BRIGHTNESS"],
                    contrast=self.cfg["DATASET"]["COLORJITTER"]["CONTRAST"],
                    saturation=self.cfg["DATASET"]["COLORJITTER"]["SATURATION"],
                    hue=self.cfg["DATASET"]["COLORJITTER"]["HUE"]
                ))
            if self.cfg["DATASET"]["GAUSSIAN_BLUR"] and self.cfg["DATASET"]["GAUSSIAN_BLUR"]["PROB"] > random.random():
                pipeline.append(transforms.GaussianBlur(
                    kernel_size=self.cfg["DATASET"]["GAUSSIAN_BLUR"]["KERNEL_SIZE"],
                    sigma=self.cfg["DATASET"]["GAUSSIAN_BLUR"]["SIGMA"]
                ))
            if self.cfg["DATASET"]["PERSPECTIVE"]:
                pipeline.append(transforms.RandomPerspective(
                    distortion_scale=self.cfg["DATASET"]["PERSPECTIVE"]["SCALE"],
                    p=self.cfg["DATASET"]["PERSPECTIVE"]["PROB"]
                ))
            if self.cfg["DATASET"]["ROTATE"] and self.cfg["DATASET"]["ROTATE"]["PROB"] > random.random():
                pipeline.append(transforms.RandomRotation(degrees=self.cfg["DATASET"]["ROTATE"]["DEGREES"]))
        if self.is_train and self.cfg["DATASET"]["RANDOM_CROP"]:
            ratio = self.cfg["DATASET"]["RANDOM_CROP"]
            pipeline.append(transforms.Resize((int(self.input_shape[0]*ratio), int(self.input_shape[1]*ratio))))
            pipeline.append(transforms.RandomResizedCrop(self.input_shape))
            safe_pipeline.append(transforms.Resize(self.input_shape))
        else:
            pipeline.append(transforms.Resize(self.input_shape))
            safe_pipeline.append(transforms.Resize(self.input_shape))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(raw_img)
        try:
            raw_img = transforms.Compose(pipeline)(raw_img)
        except Exception as e:
            print("Augmentation exception for", fp)
            print(e)
            raw_img = transforms.Compose(safe_pipeline)(raw_img)
        raw_img = np.array(raw_img)
        if len(raw_img.shape) == 2:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        if self.is_train:
            if self.cfg["DATASET"]["RANDOM_ERASING"]:
                random_erasing(raw_img,
                    p=self.cfg["DATASET"]["RANDOM_ERASING"]["PROB"],
                    scale=self.cfg["DATASET"]["RANDOM_ERASING"]["SCALE"],
                    ratio=self.cfg["DATASET"]["RANDOM_ERASING"]["RATIO"],
                    value=self.cfg["DATASET"]["RANDOM_ERASING"]["VALUE"])
        img = raw_img.astype(np.float32)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # import os
            # os.makedirs("gray", exist_ok=True)
            # cv2.imwrite(f"gray/cls{label}-{index}.jpg", img)
            img = (img/255.0 - 0.5) * 2
            img = np.expand_dims(img, axis=2)
        else:
            img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        data = torch.Tensor(img)
        return data, label, raw_img
