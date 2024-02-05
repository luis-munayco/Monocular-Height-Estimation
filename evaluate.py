import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from utils.dice_score import multiclass_dice_coeff, dice_coeff

# create a function (this my favorite choice)
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

criterion_rmse = RMSELoss

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    loss_total = 0
    loss_total_rmse=0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            #logging.info(f'mask true eval shape {mask_true.shape}')
            # predict the mask
            mask_pred = net(image)
            #logging.info(f'mask pred eval shape {mask_pred.shape}')
            # compute the regression loss (you can use L1 or L2 loss, depending on your preference)
            loss_batch = F.l1_loss(mask_pred, mask_true)  # Change this to F.l1_loss for L1 loss
            loss_total+=loss_batch
            #add evaluation with rmse
            loss_total_rmse += criterion_rmse(mask_pred, mask_true)
        loss_total = loss_total.item()
        loss_total_rmse = loss_total_rmse.item()
    net.train()
    return [loss_total / max(num_val_batches, 1), loss_total_rmse / max(num_val_batches, 1)] 
