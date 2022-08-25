import os
from collections import Counter
from .base_ds import BaseDs


class CustomDs(BaseDs):
    def __init__(self, data_path, is_train, cfg):
        super(CustomDs, self).__init__(data_path, is_train, cfg)
        for d in os.listdir(self.data_path):
            dir = os.path.join(self.data_path, d)
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
            weights = [(cls, round(c/max_cls_count)) for cls, c in weights]
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
            "GRAYSCALE": 1.0,
            "COLORJITTER": 0.1,
            "GAUSSIAN_BLUR": 0.1,
            "PERSPECTIVE": 0.1,
            "ROTATE": {
                "DEGREES": (-20, 20),
                "PROB": 0.2
            },
            "RANDOM_CROP": True
        },
        "MODEL": {
            "INPUT_SHAPE": (224, 224)
        }
    }
    ds = CustomDs(data_path, True, cfg)
    for i in range(10, 20):
        data, label, img = ds[i]
        print(data.shape)
        print(label)
        cv2.imshow("Img", img)
        cv2.waitKey()
