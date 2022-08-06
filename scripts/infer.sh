# onnx
# python infer.py \
#     --image examples/images/cat.jpeg \
#     --engine onnx \
#     --model weights/dogsvscats_shufflenetv2_none_linearcls_10eps.onnx \
#     --input-shape 224 224 \
#     --device cpu

# torch
python infer.py \
    --image examples/images/dog.jpeg \
    --engine torch \
    --device cuda:0 \
    --config configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml