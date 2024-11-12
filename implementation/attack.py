import sys, os
sys.path.append('implementation')
sys.path.append("..")
from model.model.losses import DiceLoss
from model.utils.trainer import Trainer
from model.utils.reproducibility import set_seed
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse


from metrics import StreamSegMetrics

import logging


from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.utils.distributed import make_data_sampler, make_batch_data_sampler

from core.models.model_zoo import get_segmentation_model



def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--model', type=str)
    parser.add_argument('--source_model', type=str, default='psp_resnet50_voc')
    parser.add_argument('--target_model', type=str, default='deeplabv3_resnet101_voc')
    parser.add_argument('--workers', '-j', type=int, default=0,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--pretrained_data', type=str, default='pascal_aug',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--pt', type=str, default='./../pretrained_model',
                        help='Path of pretrained model to be adversarially attacked')
    parser.add_argument('--pretrained', type=bool, default=False, help='you can train models in https://github.com/Tramac/awesome-semantic-segmentation-pytorch')
# * Dataset parameters
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')

    parser.add_argument('--data_root', default='./../datasets/voc', type=str, help='root of dataset to train/eval on')
    parser.add_argument("--download", action='store_true', default=False, help="download datasets")
    parser.add_argument('--crop_size', default=513, type=int, help='crop_size for training')
    parser.add_argument('--crop_val', action='store_true', default=True, help='To crop  val images or not')

# * Save dir
    parser.add_argument('--save', default='results', type=str, help='directory to save models')

    # * Loss
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help='Loss Criteria')
    
    # * Large transposed convolution kernels, plots and FGSM attack    
    parser.add_argument('-it', '--iterations', type=int, default=20,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-at', '--attack', type=str, default='fspgd', choices={'fspgd'},
                        help='Which adversarial attack')
    parser.add_argument('-ep', '--epsilon', type=float, default=0.03,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='number of iterations for adversarial attack')
    parser.add_argument('-nr', '--norm', type=str, default="inf", choices={'inf', 'two', 'one'},
                        help='lipschitz continuity bound to use')
    parser.add_argument('-tar', '--targeted', type=str, default="False", choices={'False', 'True'},
                        help='use a targeted attack or not')

    parser.add_argument('-m', '--mode', type=str, default='adv_attack', choices={'adv_attack', 'trans_test', 'test'},
                        help='What to do?')
    parser.add_argument('-source_layer', '--source_layer', type=str, default='layer3_2')
    parser.add_argument('-cosine', '--cosine', type=float, default=3)
    parser.add_argument('-lamda', '--lamda', type=float, default=1)

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

def get_dataset(opts, attack_root=None):
    """ Dataset And Augmentation
    """

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    if 'train' in opts.mode:
        train_dataset = get_segmentation_dataset(opts.data_root,  root=opts.data_root, split='train', mode='train', transform=data_transform, crop_size=opts.crop_size)
        train_sampler = make_data_sampler(train_dataset, False, opts.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, images_per_batch=1)

        val_dataset = get_segmentation_dataset(opts.data_root,  root=opts.data_root,split='val', mode='val', transform=data_transform, crop_size=opts.crop_size)
        val_sampler = make_data_sampler(val_dataset, False, opts.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        return train_dataset, train_batch_sampler, val_dataset, val_batch_sampler

    elif 'trans_test' == opts.mode:

        val_dataset = get_segmentation_dataset(opts.dataset, root=opts.data_root, split='val', mode='val',  transform=data_transform, crop_size=opts.crop_size,
                                               attack_root=attack_root)
        val_sampler = make_data_sampler(val_dataset, False, opts.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        return val_dataset, val_batch_sampler

    else:
        val_dataset = get_segmentation_dataset(opts.dataset,  root=opts.data_root, split='val', mode='val',  transform=data_transform, crop_size=opts.crop_size)
        val_sampler = make_data_sampler(val_dataset, False, opts.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        return val_dataset, val_batch_sampler



def main(args):
    """ device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) """
    set_seed(args.seed)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if args.iterations ==1:
        args.alpha = args.epsilon

    if (args.mode == 'trans_test') or (args.mode == 'adv_attack'):
        if args.attack == 'proposed':
            args.results_path = 'results/{}_{}_{}_{}'.format(args.attack, args.source_model, args.source_layer, args.lamda)
        else:
            args.results_path = 'results/{}_{}'.format(args.attack, args.source_model)
    else:
        args.results_path = 'results/{}'.format(args.model)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    logger = get_logger(args.results_path)

    tmp = args.targeted
    if tmp == 'True':
        args.targeted = True
    elif tmp =='False':
        args.targeted = False

    for arg, value in sorted(vars(args).items()):
        logger.info("{}: {}".format(arg, value))


    val_dataset, val_batch_sampler = get_dataset(args, attack_root=args.results_path)
    val_loader = DataLoader(dataset=val_dataset,
                                 batch_sampler=val_batch_sampler,
                                 num_workers=args.workers,
                                 pin_memory=True)


    if args.mode == 'trans_test':
        model_ = args.target_model.split('_')
    elif args.mode == 'adv_attack':
        model_ = args.source_model.split('_')
    elif args.mode == 'test':
        model_ = args.model.split('_')


    model = get_segmentation_model(model=model_[0], dataset=args.pretrained_data, backbone=model_[1],
                                    pretrained=args.pretrained, root=args.pt).to(args.device)

    criterion = {"cross_entropy": nn.CrossEntropyLoss(ignore_index=255, reduction="none"), "dice_loss": DiceLoss(reduction=None)}
    metrics = StreamSegMetrics(val_dataset.num_class)
    actual_metrics = StreamSegMetrics(val_dataset.num_class) if args.targeted else None
    initial_metrics = StreamSegMetrics(val_dataset.num_class) if args.targeted else None

    trainer = Trainer(
        model,                    # UNet model with pretrained backbone
        criterion = criterion[args.loss],
        epochs=args.epochs,               # number of epochs for model training
        metrics = metrics,
        actual_metrics = actual_metrics,
        initial_metrics = initial_metrics,
        logger = logger,
        args=args,
        num_classes=val_dataset.num_class
    )

    if args.mode == 'adv_attack':
        if args.attack == 'proposed':
            logger.info("---------- Method: {}, Source model & layer {}_{}------------".format(args.attack,args.source_model,args.source_layer))
        else:
            logger.info("---------- Method: {}, Source model-----------".format(args.attack,args.source_model))
    elif args.mode == 'trans_test':
        if args.attack == 'proposed':
            logger.info("---------- Method: {}, Source model & layer {}_{}, Target model: {}------------".format(args.attack,args.source_model,args.source_layer, args.target_model))
        else:
            logger.info("---------- Method: {}, Source model & layer {}, Target model: {}------------".format(args.attack,args.source_model, args.target_model))
    else:
        logger.info("---------- Original, Model {}------------".format(args.model))

    trainer.fit(val_loader)

    # Transferability
    args.mode = 'trans_test'
    if args.attack == 'proposed':
        logger.info(
            "---------- Method: {}, Source model & layer {}_{}, Target model: {}------------".format(args.attack,
                                                                                                     args.source_model,
                                                                                                     args.source_layer,
                                                                                                     args.target_model))
    else:
        logger.info("---------- Method: {}, Source model & layer {}, Target model: {}------------".format(args.attack,
                                                                                                          args.source_model,
                                                                                                          args.target_model))

    model_ = args.target_model.split('_')
    model = get_segmentation_model(model=model_[0], dataset=args.pretrained_data, backbone=model_[1],
                               pretrained=args.pretrained, root=args.pt).to(args.device)

    trainer.trans_test(val_loader, model=model, model_name=args.target_model)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('UNet training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    if args_.iterations == 1:
        args_.alpha = args_.epsilon
    main(args_)
