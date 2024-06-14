python export.py \
    --weights outputs/train/gender95/best.pth \
    --config outputs/train/gender95/configs.txt \
    --format onnx \
    --output outputs/train/gender95/Shufflenetv2_1x_gender_fs15_30eps_5e-4.onnx \
    --device cpu \
    --opset 11
