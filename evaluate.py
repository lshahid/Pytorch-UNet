import torch
import torch.nn.functional as F
# from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
import torch.nn as nn

from dice_score import multiclass_dice_coeff, dice_coeff
#from jaccard_index import jaccard_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, mask_threshold):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    criterion = nn.BCEWithLogitsLoss()
    criterion_loss = 0
    criterion_loss_list = []
    dice_loss_list = []
    # focal_loss_list = []
    total_loss_list = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()

                # Compute the Dice loss
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                dice_loss = 1 - (dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)).item()
                dice_loss_list.append(dice_loss)

                # Compute the Criterion loss
                criterion_loss = criterion(mask_pred.squeeze(1), mask_true.float())
                criterion_loss_list.append(criterion_loss.item())

                # # Compute the Focal loss
                # focal_l = sigmoid_focal_loss(mask_pred.squeeze(1), mask_true.float(), reduction='mean').item()
                # focal_loss_list.append(focal_l)

                # Compute the total loss
                total_loss_list.append(dice_loss + criterion_loss.item())
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    # loss_dice = 1 - (dice_score / max(num_val_batches, 1))
    # loss_criterion = criterion_loss / max(num_val_batches, 1)
    dice_score = dice_score / max(num_val_batches, 1)
    return dice_score, criterion_loss_list, dice_loss_list, total_loss_list
