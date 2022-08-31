import os

trainset_path = "data/dogs-vs-cats/train"
valset_path = "data/dogs-vs-cats/val"
classes = ("cat", "dog")
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
