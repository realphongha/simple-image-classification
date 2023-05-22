import os
from collections import Counter
from .base_ds import BaseDs


class CustomDs(BaseDs):
    def __init__(self, data_path, is_train, cfg):
        super(CustomDs, self).__init__(data_path, is_train, cfg)
        for data_path in self.data_path:
            for d in os.listdir(data_path):
                dir = os.path.join(data_path, d)
                if not os.path.isdir(dir) or d not in self.cls: continue
                for fn in os.listdir(dir):
                    if fn[-3:] not in cfg["DATASET"]["IMG_EXT"] and fn[-4:] not in cfg["DATASET"]["IMG_EXT"]: continue
                    fp = os.path.join(dir, fn)
                    self.data.append(fp)
                    self.labels.append(self.cls_dict[d])
        if cfg["DATASET"]["OVERSAMPLING"] and is_train:
            cls_count = Counter(self.labels)
            max_cls_count = max(cls_count.values())
            weights = cls_count.items()
            weights = [(cls, round(max_cls_count/c)) for cls, c in weights]
            weights = dict(weights)
            for i in range(len(self.labels)):
                lbl = self.labels[i]
                fp = self.data[i]
                for _ in range(weights[lbl]-1):
                    self.data.append(fp)
                    self.labels.append(lbl)
        lbl_counter = Counter(self.labels).items()
        lbl_counter = [(self.cls[cls], c) for cls, c in lbl_counter]
        print("Classes count:")
        print(lbl_counter)


if __name__ == "__main__":
    import cv2
    data_path = "/Users/admin/dataset/dogs-vs-cats/val"
    cfg = {
        "DATASET": {
            "CLS": "cat, dog",
            "IMG_EXT": ("jpg", "png", "jpeg"),
            "FLIP": True,
            "GRAYSCALE": 0.1,
            "COLORJITTER": {
                'PROB': 0.1,
                'BRIGHTNESS': 0.5,
                'CONTRAST': 0.5,
                'SATURATION': 0.3,
                'HUE': 0.3
            },
            "GAUSSIAN_BLUR": {
                "PROB": 0.1,
                "KERNEL_SIZE": (5, 9),
                "SIGMA": (0.1, 5.0)
            },
            "PERSPECTIVE": {
                "PROB": 0.1,
                "SCALE": 0.6
            },
            "ROTATE": {
                "DEGREES": (-20, 20),
                "PROB": 0.2
            },
            "RANDOM_CROP": 1.143,
            "RANDOM_ERASING": {
                "PROB": 0.2,
                "SCALE": (0.02, 0.33),
                "RATIO": (0.3, 3.3),
                "VALUE": 0
            },
            "OVERSAMPLING": False,
        },
        "MODEL": {
            "INPUT_SHAPE": (224, 224)
        }
    }
    ds = CustomDs(data_path, True, cfg)
    for i in range(100, 200):
        data, label, img = ds[i]
        print(data.shape)
        print(label)
        cv2.imshow("Img", img)
        cv2.waitKey()
