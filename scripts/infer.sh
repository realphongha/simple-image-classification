# onnx
# python infer.py \
#     --src image
#     --src_path examples/images/cat.jpeg \
#     --engine onnx \
#     --model weights/dogsvscats_shufflenetv2_none_linearcls_10eps.onnx \
#     --input-shape 224 224 \
#     --device cpu

# torch
# python infer.py \
#     --image examples/images/dog.jpeg \
#     --engine torch \
#     --device cuda:0 \
#     --config configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml

# with object detection
python infer.py \
    --src video --src-path path/to/video.mp4 \
    --engine onnx --input-shape 384 384 \
    --model weights/cls_weights.onnx \
    --cls cls1 cls2 cls3 \
    --det yolov5 --det-engine onnx --det-cls 0 --det-num-cls 1 --det-input 640 640 \
    --det-iou 0.55 --det-conf 0.25 \
    --det-model weights/det_weights.onnx \
    --device cpu