import torch
import torch.nn as nn
from .cls_loss import DINOLoss
from .patch_loss import ImgIBOTLoss
from .koleo_loss import KoLeoLoss


class PerALoss(nn.Module):
    def __init__(self, num_global_crops, num_local_crops, out_dim, centering='centering', dino_loss_weight=1.0, ibot_loss_weight=0.3, koleo_loss_weight=0.1, mae_loss_weight=0.01):
        super().__init__()
        self.loss_accumulator = 0
        self.dino_loss = DINOLoss(out_dim)
        self.koleo_loss = KoLeoLoss()
        # self.ibot_loss = ImgIBOTLoss(out_dim)

        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.mae_loss_weight = mae_loss_weight

        self.num_global = num_global_crops
        self.num_local = num_local_crops

        # self.local_loss_terms = max(self.num_local * self.num_global, 1)
        # self.global_loss_terms = (self.num_global - 1) * self.num_global
        self.local_loss_terms = self.num_local
        self.global_loss_terms = self.num_global
        self.centering = centering

    def l2_normalize(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + 1.e-6)**.5
        return x

    def patch_loss(self, student_patch_tokens, s_global_target):
        loss = (student_patch_tokens - s_global_target).pow(2)
        loss = loss.sum(dim=-1)
        return loss.mean()


    def forward(self, student_output, teacher_output, teacher_temp):
        s_local_cls_tokens = student_output['local_cls_tokens']
        # s_local_patch_tokens = student_output['local_patch_tokens']
        s_global_cls_tokens = student_output['global_cls_tokens']
        # s_global_patch_tokens = student_output['global_patch_tokens']
        s_global_pred_tokens = student_output['global_pred_tokens']
        s_global_target = student_output['global_patch_target']

        loss_dict = {} # for display
        loss_accumulator = 0  # for backprop

        t_cls_tokens, t_patch_tokens = self.teacher_centering(teacher_output, teacher_temp)
        # t_cls_tokens = self.teacher_centering(teacher_output, teacher_temp)


        local_dino_loss = (self.dino_loss(student_output_list=s_local_cls_tokens.chunk(self.num_local), 
                                        teacher_out_softmaxed_centered_list=t_cls_tokens[self.num_global:]) 
                                        / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["local_dino_loss"] = local_dino_loss
        loss_accumulator += self.dino_loss_weight * local_dino_loss


        global_dino_loss = (self.dino_loss(student_output_list=s_global_cls_tokens.chunk(self.num_global), 
                                        teacher_out_softmaxed_centered_list=t_cls_tokens[:self.num_global])
                                        / (self.global_loss_terms + self.local_loss_terms))
        loss_dict["global_dino_loss"] = global_dino_loss
        loss_accumulator += self.dino_loss_weight * global_dino_loss
        

        # local_ibot_loss = (self.ibot_loss(student_output_list=s_local_patch_tokens.chunk(self.num_local), 
        #                                 teacher_out_softmaxed_centered_list=t_patch_tokens[self.num_global:]) 
        #                                 / (self.global_loss_terms + self.local_loss_terms))

        # global_ibot_loss = (self.ibot_loss(student_output_list=s_global_patch_tokens.chunk(self.num_global), 
        #                                 teacher_out_softmaxed_centered_list=t_patch_tokens[:self.num_global])
        #                                 / (self.global_loss_terms + self.local_loss_terms))
        
        # ibot_loss = global_ibot_loss
        # loss_dict["ibot_loss"] = ibot_loss
        # loss_accumulator += self.ibot_loss_weight * ibot_loss


        koleo_dino_global_loss = sum(self.koleo_loss(p) for p in s_global_cls_tokens.chunk(self.num_global))
        # koleo_dino_local_loss = sum(self.koleo_loss(p) for p in s_local_cls_tokens.chunk(self.num_local))
        # koleo_ibot_global_loss = sum(self.koleo_loss(p) for p in s_global_patch_tokens.chunk(self.num_global))
        # koleo_ibot_local_loss = sum(self.koleo_loss(p) for p in s_local_patch_tokens.chunk(self.num_local))
        # koleo_loss = koleo_dino_global_loss + koleo_dino_local_loss
        loss_dict["koleo_loss"] = koleo_dino_global_loss / (self.global_loss_terms)
        loss_accumulator += self.koleo_loss_weight * koleo_dino_global_loss


        # s_patch_tokens = torch.cat([s_global_patch_tokens, s_local_patch_tokens])
        # t_patch_tokens = t_patch_tokens.view(-1, t_patch_tokens.shape[-1])
        # ibot_patch_loss = self.ibot_loss(s_patch_tokens, t_patch_tokens)
        # loss_dict["ibot_loss"] = ibot_patch_loss / (self.global_loss_terms + self.local_loss_terms)
        # loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # use MSE loss for patch tokens
        mae_loss = self.patch_loss(s_global_pred_tokens, s_global_target)
        loss_dict["mae_loss"] = mae_loss / (self.global_loss_terms * 100)
        loss_accumulator += self.mae_loss_weight * mae_loss

        return loss_accumulator, loss_dict


    def teacher_centering(self, teacher_output, teacher_temp):
        t_local_cls_tokens = teacher_output['local_cls_tokens']
        # t_local_patch_tokens = teacher_output['local_patch_tokens']
        t_global_cls_tokens = teacher_output['global_cls_tokens']
        # t_global_patch_tokens = teacher_output['global_patch_tokens']  # (num_crops * bs, num_patches, dim)

        teacher_cls_tokens = torch.cat([t_global_cls_tokens, t_local_cls_tokens])    # (num_crops * bs, dim)
        # teacher_patch_tokens = torch.cat([t_global_patch_tokens.mean(dim=1), t_local_patch_tokens.mean(dim=1)])
        # teacher_patch_tokens = torch.cat([t_global_patch_tokens, t_local_patch_tokens])
        # teacher_patch_tokens = t_global_patch_tokens


        if self.centering == "centering":
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens, teacher_temp=teacher_temp
                ).view(self.num_global + self.num_local, -1, *teacher_cls_tokens.shape[1:])   # reshape to (num_crops, bs, dim)
            self.dino_loss.update_center(teacher_cls_tokens)
            
            # teacher_ibot_softmaxed_centered_list = self.ibot_loss.softmax_center_teacher(
            #     teacher_patch_tokens, teacher_temp=teacher_temp
            #     ).view(self.num_global, -1, *teacher_patch_tokens.shape[1:])   # reshape to (num_crops, bs, dim)
            # self.ibot_loss.update_center(teacher_patch_tokens)
        else:
            raise NotImplementedError

        # return teacher_dino_softmaxed_centered_list, teacher_ibot_softmaxed_centered_list
        return teacher_dino_softmaxed_centered_list, None