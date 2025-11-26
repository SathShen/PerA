import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self, sigma):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma

    def forward(self, x, t, in_weight):
        sigma2 = self.sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        loss = (flag * (sigma2 / 2.) * (diff ** 2) +
                (1 - flag) * (abs_diff - 0.5 / sigma2))
        return loss.sum()

class LocLoss(nn.Module):
    def __init__(self, sigma):
        super(LocLoss, self).__init__()
        self.sigma = sigma
        self.smooth_l1_loss = SmoothL1Loss(sigma)

    def forward(self, pred_loc, gt_loc, gt_label):
        in_weight = torch.zeros(gt_loc.shape).cuda()
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike the original implementation, 
        # we don't need inside_weight and outside_weight; they can be calculated from gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
        loc_loss = self.smooth_l1_loss(pred_loc, gt_loc, in_weight.detach())
        # Normalize by the total number of negative and positive rois.
        loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
        return loc_loss

