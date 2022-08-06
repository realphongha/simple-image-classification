# Simple image classification
Simple image classification with Pytorch. Currently supporting:
## Backbones:
* ShuffleNet V2
* ShuffleNet V2 Plus
* MobileNet V3
## Neck:
* B-CNN
## Head:
* Linear 
## Loss function:
* Cross entropy
## Augmentation:
* Horizontal flip
* Grayscale
* Colorjitter
* Gaussian blur
* Perspective
* Rotate
* Random crop
## Convert and run in engine:
* ONNX
## to be added more soon...

# Tutorial
## Install:
Clone this repo:  
`git clone https://github.com/realphongha/simple-image-classification.git`  
Go in the repo and install requirements:  
`cd simple-image-classification`  
`pip install -r requirements.txt`
## Training:
Take the training process of the [Dogs vs Cats dataset](https://www.kaggle.com/competitions/dogs-vs-cats/code) as an example.
### Prepare dataset:
Your dataset should be like this:  
<pre>
simple-image-classification/  
  data/
    your_dataset/
      train/
        label1/
          a.jpg
          b.jpg
        label2/
          c.jpg
        ...
      val/
        label1/
          d.jpg
          e.jpg
        label2/
          f.jpg
        ...
</pre>
First `cd simple-image-classification` and `mkdir data && mkdir pretrained && cd data`  
Go [here](https://drive.google.com/file/d/1fYzcZ1scMwrDriqpxNOJCF5gg1l9d__Z/view?usp=sharing) to get the Dogs vs cats dataset that is already rearranged like that. Unzip and put `dogs-vs-cats` in `data`.  
Go [here](https://drive.google.com/file/d/1APmyeJ0uN8zju3dSJmklItNOf3OkExzw/view?usp=sharing) to get the pretrained weights for ShuffleNet V2 on ImageNet and put it in `pretrained`.
### Run the command below to start training:
`python train.py --config configs/customds/configs/customds/dogsvscats_shufflenetv2_none_linearcls_10eps.yaml` 

## Evaluating:
Go to config file and specify path to model weights at `TEST.WEIGHTS` and run:  
`python val.py --config path/to/config.yaml`  
(Remember to set up dataset path as well).

## Running inference:
See `./scripts/infer.sh` and `infer.py`.

## Exporting models:
See `./scripts/export.sh` and `export.py`.
