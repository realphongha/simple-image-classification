import os

trainset_path = "data/dogs-vs-cats/train"
valset_path = "data/dogs-vs-cats/val"
trainset_path = "/mnt/hdd18tb/projects/phonghh/datasets/age_gender/fs15_age_cls/train"
valset_path = "/mnt/hdd18tb/projects/phonghh/datasets/age_gender/fs15_age_cls/val"
classes = ("young", "mid", "old")
exts = ("jpg", "jpeg", "png")

print("Trainset:")
all = 0
for cls in classes:
    path = os.path.join(trainset_path, cls)
    count = 0
    for fn in os.listdir(path):
        for ext in exts:
            if fn.endswith(ext):
                count += 1
                break
    all += count
    print("  %s: %i images" % (cls, count))
print("Total: %i images" % all)

print("Valset:")
all = 0
for cls in classes:
    path = os.path.join(valset_path, cls)
    count = 0
    for fn in os.listdir(path):
        for ext in exts:
            if fn.endswith(ext):
                count += 1
                break
    all += count
    print("  %s: %i images" % (cls, count))
print("Total: %i images" % all)
