from .shufflenetv2 import get_shufflenetv2
from .shufflenetv2_plus import get_shufflenetv2_plus
from .mobilenetv3 import get_mobilenet_v3
from .mobileone import get_mobileone


def build_backbone(cfg, training, model):
    backbone_name = cfg["MODEL"]["BACKBONE"]["NAME"]
    if backbone_name == "shufflenetv2":
        backbone = get_shufflenetv2(
            cfg["MODEL"]["BACKBONE"]["WIDEN_FACTOR"],
            cfg["TRAIN"]["PRETRAINED"] if training else None
        )
        out_channels = backbone.stage_out_channels[-1]
    elif backbone_name == "shufflenetv2_plus":
        backbone = get_shufflenetv2_plus(
            cfg["MODEL"]["INPUT_SHAPE"],
            cfg["DATASET"]["NUM_CLS"],
            cfg["MODEL"]["BACKBONE"]["WIDEN_FACTOR"],
            cfg["TRAIN"]["PRETRAINED"] if training else None
        )
        out_channels = backbone.stage_out_channels[-1]
    elif backbone_name == "mobilenetv3":
        backbone, out_channels = get_mobilenet_v3(
            cfg["MODEL"]["BACKBONE"]["WIDEN_FACTOR"],
            cfg["TRAIN"]["PRETRAINED"] if training else None
        )
    elif backbone_name == "mobileone":
        backbone = get_mobileone(
            cfg["DATASET"]["NUM_CLS"],
            training,
            cfg["MODEL"]["BACKBONE"]["WIDEN_FACTOR"],
            cfg["TRAIN"]["PRETRAINED"] if training else None
        )
        out_channels = backbone.in_planes
    else:
        raise NotImplementedError("Backbone %s is not implemented!" %
            backbone_name)
    return backbone, out_channels
