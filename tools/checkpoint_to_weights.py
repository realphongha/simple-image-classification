"""
Convert .pth checkpoint (with epoch and weights information) to weights only.
"""
import torch


ckpt_file = "path/to/ckpt"
output_file = "path/to/output"
device = "cpu"
checkpoint = torch.load(ckpt_file, map_location=device)
state_dict = checkpoint['state_dict']
torch.save(state_dict, output_file)
