import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


"""
preds: Batchsize,Classes,Height,Width
labels: Batchsize,Classes,Height,Width

preds contains possibilities for every class and every points.
labels should be one-hot-encoded.
"""


class SoftIoULoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super().__init__()
        self.size_average = size_average

    def forward(self, preds, targets):
        eps = 1e-6
        preds = torch.sigmoid(preds)    # N,Class,W,H    每个点样本有class个概率，每个独热编码样本点有class-1个0和一个1

        inter = preds * targets
        inter = inter.sum(dim=(2, 3))

        union = preds + targets - (preds * targets)
        union = union.sum(dim=(2, 3))

        loss = 1 - inter / (union + eps)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super().__init__()
        self.size_average = size_average

    def forward(self, preds, targets):
        eps = 1
        inter = preds * targets
        dice_loss = (2. * (inter.sum(dim=(2, 3))) + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
        if self.size_average:
            return 1 - dice_loss.mean()
        else:
            return 1 - dice_loss.sum()


class DiceBCELoss(nn.Module):
    def __init__(self, size_average=True, a=0.8, b=0.2, ignore_index=-100):
        super().__init__()
        self.size_average = size_average
        self.BCE = BCELoss()
        self.Dice = DiceLoss()
        self.a = a
        self.b = b

    def forward(self, preds, targets):
        return self.a * self.BCE(preds, targets) + self.b * self.Dice(preds, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, preds, targets):
        # sig_inputs = torch.sigmoid(inputs)

        ce_loss = F.cross_entropy(preds, targets.float(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class CrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, preds, targets):
        # sig_inputs = torch.sigmoid(inputs)
        ce_loss = F.cross_entropy(preds, targets.float(), reduction='none', ignore_index=self.ignore_index)
        if self.size_average:
            return ce_loss.mean()
        else:
            return ce_loss.sum()


class BCELoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=-100):
        super(BCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy(preds, targets.float(), reduction='none')
        if self.size_average:
            return bce_loss.mean()
        else:
            return bce_loss.sum()


def loss_test(loss, inputs, targets):
    print("input:\n", inputs)
    print("target:\n", targets)
    output = loss(inputs, targets)
    print(output)
    loss2 = BCELoss()
    output2 = loss2(inputs, targets)
    print(output2)


if __name__ == '__main__':
    inputs = torch.rand(4, 2, 5, 5, dtype=torch.float)  # Batch_size, Class, H, W
    targets = torch.randint(0, 2, (4, 2, 5, 5))  # Batch_size, H, W
    # label_idx_list = list(range(2))
    label_idx_list = [0, 1]
    loss_test(DiceLoss(), inputs, targets)
