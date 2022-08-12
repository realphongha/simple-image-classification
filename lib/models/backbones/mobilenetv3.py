import torch.nn as nn
import torchvision.models as models
from ..utils import load_checkpoint


def get_mobilenet_v3(size="small", pretrained="imagenet"):
    imagenet_pretrained = (pretrained == "imagenet")
    if size == "small":
        model = models.mobilenet_v3_small(pretrained=imagenet_pretrained)
    elif size == "large":
        model = models.mobilenet_v3_large(pretrained=imagenet_pretrained)
    model = nn.Sequential(*(list(model.children())[:-2]))
    
    if not imagenet_pretrained and pretrained:
        load_checkpoint(model, pretrained, strict=False)

    for param in model.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    import torch 
    
    model = get_mobilenet_v3("small", "imagenet")
    print(model)
    print("Params:", sum(p.numel() for p in model.parameters()))
    
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
    