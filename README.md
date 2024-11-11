# FSPGD

## Abstract
Transferability, the ability of adversarial examples crafted for one model to deceive other models, is crucial for black-box attacks. Despite advancements in attack methods for semantic segmentation, transferability remains limited, reducing their effectiveness in real-world applications. To address this, we introduce the Feature Similarity Projected Gradient Descent (FSPGD) attack, a novel black-box approach that enhances both attack performance and transferability. Unlike conventional segmentation attacks that rely on output predictions for gradient calculation, FSPGD computes gradients from intermediate layer features. Specifically, our method introduces a loss function that targets local information by comparing features between clean and attacked images, while also disrupting contextual information through spatial relationships. Experiments on the Pascal VOC 2012 and Cityscapes datasets demonstrate that FSPGD achieves superior transferability and attack performance, establishing a new state-of-the-art benchmark.
## Implementing
## 1. Installation

```
conda create -n fspgd python =3.8
conda activate fspgd
pip install -r requirements.txt
```
## 2  Preparation
### Pascal VOC 2012
```
/dataset
    /voc 
        /SegmentationClass
        /JPEGImages
        ...
```
### Cityscapes
```
/dataset
    /citys 
        /gtFine
        /leftImg8bit
```
## 3. Run
```
cd implementation
python attack.py --attack fspgd --mode adv_attack --dataset pascal_voc --pretrained_data pascal_aug  --cosine 3 --source_model deeplabv3_resnet50 --target_model psp_resnet101
```
or, you can use bash script [attack.sh](implementation/attack.sh)


### Partial code are from

[1] [CosPGD](https://github.com/shashankskagnihotri/cospgd) 

[2] [Semantic Segmentation on Pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)


