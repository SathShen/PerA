import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=-100):
        super(CDCrossEntropyLoss, self).__init__()
        """
        Args:
            num_classes (int): The number of classes
            ignore_index (int): The index of the ignore class, e.g. 0 for not change
        """
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, preds, targets):
        if self.num_classes == 2:
            ce_loss = F.cross_entropy(preds, targets.float(), reduction='mean', ignore_index=self.ignore_index)
            return ce_loss
        else:
            predsA, predsB, predsmask = preds
            targetsA, targetsB, targetsmask = targets
            ce_lossA = F.cross_entropy(predsA, targetsA.float(), reduction='mean', ignore_index=self.ignore_index)
            ce_lossB = F.cross_entropy(predsB, targetsB.float(), reduction='mean', ignore_index=self.ignore_index)
            ce_lossmask = F.cross_entropy(predsmask, targetsmask.float(), reduction='mean', ignore_index=self.ignore_index)
            loss_ret = (ce_lossA + ce_lossB + ce_lossmask) / 3
            return loss_ret


class CDBCELoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=-100):
        super(CDBCELoss, self).__init__()
        """
        Args:
            num_classes (int): The number of classes
            ignore_index (int): The index of the ignore class, e.g. 0 for not change
        """
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        if self.num_classes > 2:
            KeyError("CDBCELoss only support binary classification")

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='mean')
        return bce_loss




class CDFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, ignore_index=-100):
        super(CDFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        if self.num_classes == 2:
            ce_loss = F.cross_entropy(preds, targets.float(), reduction='none', ignore_index=self.ignore_index)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
        else:
            predsA, predsB, predsmask = preds
            targetsA, targetsB, targetsmask = targets
            ce_lossA = F.cross_entropy(predsA, targetsA.float(), reduction='none', ignore_index=self.ignore_index)
            ce_lossB = F.cross_entropy(predsB, targetsB.float(), reduction='none', ignore_index=self.ignore_index)
            ce_lossmask = F.cross_entropy(predsmask, targetsmask.float(), reduction='none', ignore_index=self.ignore_index)
            ptA = torch.exp(-ce_lossA)
            ptB = torch.exp(-ce_lossB)
            ptmask = torch.exp(-ce_lossmask)
            focal_lossA = self.alpha * (1-ptA)**self.gamma * ce_lossA
            focal_lossB = self.alpha * (1-ptB)**self.gamma * ce_lossB
            focal_lossmask = self.alpha * (1-ptmask)**self.gamma * ce_lossmask
            loss_ret = (focal_lossA + focal_lossB + focal_lossmask) / 3
            return loss_ret.mean()