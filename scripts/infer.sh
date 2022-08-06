python infer.py \
    --image examples/images/cat.jpeg \
    --engine onnx \
    --model weights/dogsvscats_shufflenetv2_none_linearcls_10eps.onnx \
    --input-shape 224 224 \
    --device cpu