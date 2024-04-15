import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/test_dataset/imgs/')
dir_mask = Path('./data/test_dataset/masks/')
dir_checkpoint = Path('./checkpoints/')


def test_model(
        model,
        device,
        batch_size: int = 5,
        img_scale: float = 0.5,
        amp: bool = False
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_test = len(dataset)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=True) #, num_workers=os.cpu_count()
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)

    # (Initialize logging)

    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Training size:   {n_test}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    
    # Begin the test
    test_score = evaluate(model, test_loader, device, amp)
    #scheduler.step(val_score)

    logging.info('Validation L1 Loss: {}'.format(test_score[0]))# L1
    logging.info('Validation RMSE Loss: {}'.format(test_score[1]))# RMSE

def get_args():
    #check errors
    parser = argparse.ArgumentParser(description='Test the UNet on images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #remove 4th channel, no data
    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        test_model(
            model=model,
            batch_size=args.batch_size,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        test_model(
            model=model,
            batch_size=args.batch_size,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
