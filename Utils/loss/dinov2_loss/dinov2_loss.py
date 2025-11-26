import torch
import torch.nn as nn
from .dino_clstoken_loss import DINOSmucLoss, DINOLoss
from .koleo_loss import KoLeoLoss
from .ibot_patch_loss import iBOTPatchLoss

class DinoV2SmucLoss(nn.Module):
    def __init__(self, num_global_crops, num_local_crops, out_dim, centering='centering', dino_loss_weight=1.0, ibot_loss_weight=1.0, koleo_loss_weight=0.1):
        super().__init__()
        self.loss_accumulator = 0
        self.dino_loss = DINOSmucLoss(out_dim)
        self.koleo_loss = KoLeoLoss()
        self.ibot_loss = iBOTPatchLoss(out_dim)

        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight

        self.num_global = num_global_crops
        self.num_local = num_local_crops

        # self.local_loss_terms = max(self.num_local * self.num_global, 1)
        # self.global_loss_terms = (self.num_global - 1) * self.num_global
        self.local_loss_terms = self.num_local
        self.global_loss_terms = self.num_global
        self.centering = centering


    def forward(self, student_output, teacher_output, teacher_temp, images):
        loss_dict = {} # for display
        loss_accumulator = 0  # for backprop
        loss_scales = self.num_global  # this is here since we process global crops together
        ibot_loss_scale = 1.0 / self.num_global

        t_dino_cls_tokens_list, t_ibot_patch = self.teacher_centering(teacher_output, teacher_temp, images['num_masked'])

        s_local_cls_tokens = student_output['local_cls_tokens']
        s_global_cls_tokens = student_output['global_cls_tokens']
        s_global_patch_tokens = student_output['global_patch_tokens']

        local_dino_loss = (self.dino_loss(student_output_list=s_local_cls_tokens.chunk(self.num_local), 
                                        teacher_out_softmaxed_centered_list=t_dino_cls_tokens_list[self.num_global:],) 
                                        / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["local_dino_loss"] = local_dino_loss
        loss_accumulator += self.dino_loss_weight * local_dino_loss

        global_dino_loss = (self.dino_loss(student_output_list=[s_global_cls_tokens], teacher_out_softmaxed_centered_list=
                                           [t_dino_cls_tokens_list[:self.num_global].flatten(0, 1)]) * loss_scales 
                                           / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["global_dino_loss"] = global_dino_loss
        loss_accumulator += self.dino_loss_weight * global_dino_loss

        
        koleo_loss = sum(self.koleo_loss(p) for p in s_global_cls_tokens.chunk(self.num_global))
        loss_dict["koleo_loss"] = (koleo_loss / loss_scales)
        loss_accumulator += self.koleo_loss_weight * koleo_loss
        

        ibot_patch_loss = (self.ibot_loss.forward_masked(s_global_patch_tokens, t_ibot_patch, student_masks_flat=images["masks"], 
                            n_masked_patches=images["num_masked"], masks_weight=images["masks_weight"]) * loss_scales * ibot_loss_scale)
        loss_dict["ibot_loss"] = ibot_patch_loss / loss_scales
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss


        return loss_accumulator, loss_dict


    def teacher_centering(self, teacher_output, teacher_temp, num_masked):
        teacher_cls_tokens_after_head = torch.cat([teacher_output['global_cls_tokens'], teacher_output['local_cls_tokens']])
        masked_teacher_patch_tokens_after_head = teacher_output['global_patch_tokens']
        if self.centering == "centering":
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(self.num_global + self.num_local, -1, *teacher_cls_tokens_after_head.shape[1:])
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            
            masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
            masked_teacher_ibot_softmaxed_centered = self.ibot_loss.softmax_center_teacher(
                masked_teacher_patch_tokens_after_head[:, :num_masked.item()], teacher_temp=teacher_temp
            )
            masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
            self.ibot_loss.update_center(masked_teacher_patch_tokens_after_head[:num_masked.item()])

        elif self.centering == "sinkhorn_knopp":
            teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(self.num_global + self.num_local, -1, *teacher_cls_tokens_after_head.shape[1:])

            masked_teacher_ibot_softmaxed_centered = self.ibot_loss.sinkhorn_knopp_teacher(
                masked_teacher_patch_tokens_after_head,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=num_masked)
        else:
            raise NotImplementedError

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered
    




class DinoV2Loss(nn.Module):
    def __init__(self, num_global_crops, num_local_crops, out_dim, centering='centering', dino_loss_weight=1.0, ibot_loss_weight=1.0, koleo_loss_weight=0.1):
        super().__init__()
        self.loss_accumulator = 0
        self.dino_loss = DINOLoss(out_dim)
        self.koleo_loss = KoLeoLoss()
        self.ibot_loss = iBOTPatchLoss(out_dim)

        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight

        self.num_global = num_global_crops
        self.num_local = num_local_crops

        self.local_loss_terms = max(self.num_local * self.num_global, 1)
        self.global_loss_terms = (self.num_global - 1) * self.num_global
        self.centering = centering


    def forward(self, student_output, teacher_output, teacher_temp, images):
        loss_dict = {} # for display
        loss_accumulator = 0  # for backprop
        loss_scales = self.num_global  # this is here since we process global crops together
        ibot_loss_scale = 1.0 / self.num_global

        t_dino_cls_list, t_ibot_patch = self.teacher_centering(teacher_output, teacher_temp, images['num_masked'])

        s_local_cls_tokens = student_output['local_cls_tokens']
        s_global_cls_tokens = student_output['global_cls_tokens']
        s_global_patch_tokens = student_output['global_patch_tokens']

        
        local_dino_loss = (self.dino_loss(student_output_list=s_local_cls_tokens.chunk(self.num_local), 
                                        teacher_out_softmaxed_centered_list=t_dino_cls_list,) / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["local_dino_loss"] = local_dino_loss
        loss_accumulator += self.dino_loss_weight * local_dino_loss


        global_dino_loss = (self.dino_loss(student_output_list=[s_global_cls_tokens], teacher_out_softmaxed_centered_list=
                                           [t_dino_cls_list.flatten(0, 1)]) * loss_scales 
                                           / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["global_dino_loss"] = global_dino_loss
        loss_accumulator += self.dino_loss_weight * global_dino_loss


        koleo_loss = sum(self.koleo_loss(p) for p in s_global_cls_tokens.chunk(self.num_global))
        loss_dict["koleo_loss"] = (koleo_loss / loss_scales)
        loss_accumulator += self.koleo_loss_weight * koleo_loss
        

        ibot_patch_loss = (self.ibot_loss.forward_masked(s_global_patch_tokens, t_ibot_patch, student_masks_flat=images["masks"], 
                            n_masked_patches=images["num_masked"], masks_weight=images["masks_weight"]) * loss_scales * ibot_loss_scale)
        loss_dict["ibot_loss"] = ibot_patch_loss / loss_scales
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss


        return loss_accumulator, loss_dict


    def teacher_centering(self, teacher_output, teacher_temp, num_masked):
        teacher_cls_tokens_after_head = teacher_output['global_cls_tokens']
        masked_teacher_patch_tokens_after_head = teacher_output['global_patch_tokens']
        if self.centering == "centering":
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
            ).view(self.num_global, -1, *teacher_cls_tokens_after_head.shape[1:])
            self.dino_loss.update_center(teacher_cls_tokens_after_head)
            
            masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
            masked_teacher_ibot_softmaxed_centered = self.ibot_loss.softmax_center_teacher(
                masked_teacher_patch_tokens_after_head[:, :num_masked.item()], teacher_temp=teacher_temp
            )
            masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
            self.ibot_loss.update_center(masked_teacher_patch_tokens_after_head[:num_masked.item()])

        elif self.centering == "sinkhorn_knopp":
            teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                teacher_cls_tokens_after_head, teacher_temp=teacher_temp
            ).view(self.num_global, -1, *teacher_cls_tokens_after_head.shape[1:])

            masked_teacher_ibot_softmaxed_centered = self.ibot_loss.sinkhorn_knopp_teacher(
                masked_teacher_patch_tokens_after_head,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=num_masked)
        else:
            raise NotImplementedError

        return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered