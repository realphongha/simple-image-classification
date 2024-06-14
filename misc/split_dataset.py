import os
import shutil
import random
from tqdm import tqdm


SPLIT_RATIO = 0.8
INPUT_PATH = "/mnt/hdd18tb/projects/phonghh/datasets/age_gender/fs15_gender_cls_original/"
OUTPUT_PATH = "/mnt/hdd18tb/projects/phonghh/datasets/age_gender/fs15_gender_cls/"
CLASSES = ["female", "male"]
EXTS = ["jpg"]
SHUFFLE = True
SEED = 42

images = dict()
output_images = dict()
for cls in CLASSES:
    images[cls] = list()
    output_images[cls] = dict()

for cls in os.listdir(INPUT_PATH):
    if cls not in CLASSES: continue
    fdp = os.path.join(INPUT_PATH, cls)
    for fn in os.listdir(fdp):
        ext = fn.split(".")[-1]
        if ext not in EXTS: continue
        fp = os.path.join(fdp, fn)
        images[cls].append(fp)

for cls in images:
    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(images[cls])
    all_len = len(images[cls])
    output_images[cls]["train"] = images[cls][:int(all_len*SPLIT_RATIO)]
    output_images[cls]["val"] = images[cls][int(all_len*SPLIT_RATIO):]

for cls in tqdm(output_images):
    for dset in ("train", "val"):
        target_dir = os.path.join(OUTPUT_PATH, dset, cls)
        os.makedirs(target_dir, exist_ok=True)
        for fp in output_images[cls][dset]:
            shutil.copy(fp, target_dir)
