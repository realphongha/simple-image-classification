import os
import torch


def save_checkpoint(states, is_best, output_dir):
    print("Saving checkpoint to %s..." % os.path.join(output_dir, "last.pth"))
    torch.save(states, os.path.join(output_dir, "last.pth"))
    if is_best:
        print("Saving best checkpoint to %s..." % os.path.join(output_dir, "best.pth"))
        torch.save(states, os.path.join(output_dir, "best.pth"))
