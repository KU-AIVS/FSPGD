# FSPGD


<img src="https://github.com/user-attachments/assets/30589627-5d48-4b54-afc4-7397f312be18"  width="800"/>






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
### 2.1 Pretrained model
we trained models(pspnet-res50, pspnet-res101, deeplabv3-res50, deeplabv3-res101, fcn-vgg16) using the code from this [site](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
If the trained models, move the model file to the 'pretrained_model' folder 
```
/pretrained_model
    /deeplabv3_resnet50_voc.pth
    /deeplabv3_resnet101_voc.pth
    /psp_resnet50_voc.pth
    /psp_resnet101_voc.pth
```
### 2.2 Dataset
####  Pascal VOC 2012
Download  'training/validation data' file from [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) and extract it to 'dataset/voc'.

```
/datasets
    /voc
        /VOC2012
            /Annotation
            /ImageSets
            /JPEGImages
            /SegmentationClass
            ...
```
Download 'gtFine_trainvaltest.zip' and 'leftImg8bit_trainvaltest.zip' from [Cityscapes](https://www.cityscapes-dataset.com/) 
and extract it to dataset/citys
#### Cityscapes
```
/datasets
    /citys 
        /gtFine
        /leftImg8bit
```

## 3. Run
<img src="https://github.com/user-attachments/assets/a8a75169-6a7e-4dc2-be2a-84add7903bbd"  width="800"/>


Generate adversarial examples from the proposed attack method and evaluate transferability.

```
cd implementation
python attack.py --attack fspgd --mode adv_attack --dataset pascal_voc --pretrained_data pascal_aug --cosine 3 --source_model psp_resnet50_voc --target_model deeplabv3_resnet101_voc
# if you have pretrained model, you have to change '--pretrained' True(default False)
```
or, you can use bash script [attack.sh](implementation/attack.sh)

<img src="https://github.com/user-attachments/assets/7f90adab-da05-492c-bc0d-b069896a24ac"  width="800"/>


### Partial code are from

[1] [CosPGD](https://github.com/shashankskagnihotri/cospgd) 

[2] [Semantic Segmentation on Pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)


