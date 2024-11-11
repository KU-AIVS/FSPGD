import sys
sys.path.append('unet_backbones')
from backbones_unet.model.unet import Unet
from backbones_unet.utils.dataset import SemanticSegmentationDataset
from backbones_unet.model.losses import DiceLoss
from backbones_unet.utils.trainer import Trainer
from backbones_unet.utils.reproducibility import set_seed
from cospgd import functions

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import torchvision
from torchvision.transforms import Normalize

from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et

from metrics import StreamSegMetrics

import json
import logging
import numpy as np

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--small_decoder', default="True", type=str)
    parser.add_argument('-en', '--encoder', type=str, default='resnet50',
                        help="list of all models: { 'vgg16', 'vgg16_bn','resnet50', 'resnet50_gn', 'resnet50d','resnet101', 'resnet101d'}")
# * Dataset parameters
    parser.add_argument('--dataset', default='pascalvoc2012', type=str, help='dataset to train/eval on like pascalvoc2012 or cityscapes. Dataset PASCAL VOC2012 is currently not supported')
    parser.add_argument("--download", action='store_true', default=False, help="download datasets")
    parser.add_argument('--crop_size', default=513, type=int, help='crop_size for training')
    parser.add_argument('--crop_val', action='store_true', default=False, help='To crop  val images or not')
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

# * Save dir
    parser.add_argument('--save', default='experiments', type=str, help='directory to save models')   
    

    # * Loss
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='Loss Criteria')
    
    # * Large transposed convolution kernels, plots and FGSM attack    
    parser.add_argument('-it', '--iterations', type=int, default=50000,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-at', '--attack', type=str, default='fgsm', choices={'fgsm', 'cospgd', 'segpgd', 'pgd'},
                        help='Which adversarial attack')
    parser.add_argument('-ep', '--epsilon', type=float, default=0.03,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-nr', '--norm', type=str, default="inf", choices={'inf', 'two', 'one'},
                        help='lipschitz continuity bound to use')
    parser.add_argument('-tar', '--targeted', type=str, default="False", choices={'False', 'True'},
                        help='use a targeted attack or not')    
    parser.add_argument('-m', '--mode', type=str, default='test', choices={'adv_attack', 'adv_train', 'train', 'test'},
                        help='What to do?')

    return parser


def get_logger(save_folder):
    log_path = str(save_folder) + '/log.log'
    logging.basicConfig(filename=log_path, filemode='a')
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'pascalvoc2012':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtToTensor(),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtResize( opts.crop_size ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize( opts.crop_size ),
            et.ExtToTensor(),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def main(args):
    """ device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) """
    set_seed(args.seed)
    dataset_path={"pascalvoc2012": {"num_classes":21, "data_root": "datasets/data/VOCdevkit/VOC2012", "crop_size":args.crop_size},
            "cityscapes": {"num_classes":19, "data_root": "datasets/data/cityscapes", "crop_size":args.crop_size}}

    args.data_root = dataset_path[args.dataset]["data_root"] 
    args.crop_size = dataset_path[args.dataset]["crop_size"] 
    args.num_classes = dataset_path[args.dataset]["num_classes"] 

    tmp_decoder = args.small_decoder
    if tmp_decoder == "True":
        args.small_decoder = True
    elif tmp_decoder == "False":
        args.small_decoder = False

    save_path = os.path.join(args.save, args.dataset, args.encoder, "small_deocder_"+str(args.small_decoder), "epochs_"+str(args.epochs), "lr_"+str(args.lr), 'seed_'+str(args.seed))
    args.save_path = save_path
    model_path = os.path.join(save_path , "model")
    json_path = os.path.join(save_path, "losses.json")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(args.save_path)

    

    # create a torch.utils.data.Dataset/DataLoader
    train_img_path = 'example_data/train/images' 
    train_mask_path = 'example_data/train/masks'

    val_img_path = 'example_data/val/images' 
    val_mask_path = 'example_data/val/masks'

    """ train_img_path = dataset_path[args.dataset]["train_img_path"]
    train_mask_path = dataset_path[args.dataset]["train_mask_path"]
    val_img_path = dataset_path[args.dataset]["val_img_path"]
    val_mask_path = dataset_path[args.dataset]["val_mask_path"]"""

    

    for arg, value in sorted(vars(args).items()):
        logger.info("{}: {}".format(arg, value))

    #train_dataset = SemanticSegmentationDataset(train_img_path, train_mask_path)
    #val_dataset = SemanticSegmentationDataset(val_img_path, val_mask_path)
    train_dataset, val_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=24, shuffle=True)

    model = Unet(
        #backbone='convnext_base', # backbone network name
        backbone=args.encoder,
        small_decoder = args.small_decoder,
        in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
        num_classes=dataset_path[args.dataset]["num_classes"],
        #num_classes=1,            # output channels (number of classes in your dataset)
    )

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n\n\n\t\t\tNUMBER OF PARAMETERS: {}".format(num_params))

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    model = nn.Sequential(Normalize(mean = mean, std = std), model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    criterion = {"cross_entropy": nn.CrossEntropyLoss(ignore_index=255), "dice_loss": DiceLoss()} 

    metrics = StreamSegMetrics(dataset_path[args.dataset]["num_classes"])

    trainer = Trainer(
        model,                    # UNet model with pretrained backbone
        criterion = criterion[args.loss],
        #criterion=DiceLoss(),     # loss function for model convergence
        optimizer=optimizer,      # optimizer for regularization
        epochs=args.epochs,               # number of epochs for model training
        metrics = metrics,
        args=args,
        logger = logger,
        model_save_path = os.path.join(model_path, "best_model.pt")
    )

    trainer.fit(train_loader, val_loader)        
    
    torch.save({"epoch": args.epochs, "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": trainer.train_losses_,
                "val_loss": trainer.val_losses_}, 
                os.path.join(model_path, "final_model.pt"))

    losses = {"train loss": trainer.train_losses_.detach().cpu().tolist(), 
                "val loss": trainer.val_losses_.detach().cpu().tolist(), 
                "score": trainer.metrics.get_results()}
    json_losses = json.dumps(losses, indent=4)
    with open(json_path, "w") as f:
        f.write(json_losses)
    
    
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser('UNet training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    if args_.iterations == 1:
        args_.alpha = args_.epsilon
    main(args_)