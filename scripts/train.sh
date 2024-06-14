# CUDA_VISIBLE_DEVICES=0 \
#     python train.py --config configs/customds/vit224b32_gender_lagenda_10eps.yaml --compile default
CUDA_VISIBLE_DEVICES=0 \
    python train.py --config configs/customds/vit224b16_age_lagenda_10eps.yaml --compile default
# CUDA_VISIBLE_DEVICES=0 \
#     python train.py --config configs/customds/gender_fs15_10eps.yaml
# CUDA_VISIBLE_DEVICES=0 \
#     python train.py --config configs/customds/age_lagenda_10eps.yaml
# CUDA_VISIBLE_DEVICES=0 \
#     python train.py --config configs/customds/age_fs15_10eps.yaml
