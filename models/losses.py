import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.float()
        bce = self.bce(pred, target)
        inter = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return bce + dice.mean()


def rtr_loss(mask_pred, skel_pred, mask_gt, skel_gt, skel_weight=0.5):
    criterion = BCEDiceLoss()
    loss_mask = criterion(mask_pred, mask_gt)
    loss_skel = criterion(skel_pred, skel_gt)
    return loss_mask + skel_weight * loss_skel
