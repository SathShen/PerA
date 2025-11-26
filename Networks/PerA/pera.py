import torch
import torch.nn.functional as F
from .layers import PerA_cls_head, PerA_patch_head, PerA_pixel_head
from Networks.Pretrain.DinoV2.vision_transformer import DinoV2VisionTransformer
from torch import nn


class ModelWrapper(nn.Module):
    def __init__(self, backbone, cls_head, patch_head):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.patch_head = patch_head


class PerA(nn.Module):
    def __init__(self, cfg):
        super(PerA, self).__init__()
        self.cfg = cfg
        student_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                            patch_size=cfg.NET.PATCH_SIZE, 
                                            embed_dim=cfg.NET.EMBED_DIM,
                                            depth=cfg.NET.DEPTH,
                                            num_heads=cfg.NET.NUM_HEADS,
                                            mlp_ratio=cfg.NET.MLP_RATIO,
                                            drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                            drop_path_uniform=True)
        student_backbone.mask_token = None
        teacher_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                            patch_size=cfg.NET.PATCH_SIZE, 
                                            embed_dim=cfg.NET.EMBED_DIM,
                                            depth=cfg.NET.DEPTH,
                                            num_heads=cfg.NET.NUM_HEADS,
                                            mlp_ratio=cfg.NET.MLP_RATIO)
        teacher_backbone.mask_token = None
        S_cls_head = PerA_cls_head(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        T_cls_head = PerA_cls_head(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        # S_patch_head = PerA_patch_head(in_dim=cfg.NET.EMBED_DIM, 
        #                             out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
        #                             use_bn=False, 
        #                             nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
        #                             hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
        #                             bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM
        #                           )
        # T_patch_head = PerA_patch_head(in_dim=cfg.NET.EMBED_DIM, 
        #                             out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
        #                             use_bn=False, 
        #                             nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
        #                             hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
        #                             bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM
        #                           )
        
        self.pred_head = PerA_pixel_head(in_dim=cfg.NET.EMBED_DIM, 
                                    out_dim=cfg.NET.PATCH_SIZE ** 2 * 3,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM
                                  )

        self.s_ratio = cfg.NET.PERA.S_RATIO
        self.t_ratio = cfg.NET.PERA.T_RATIO

        self.s_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.NET.EMBED_DIM))
        torch.nn.init.normal_(self.s_mask_token, std=.02)


        self.student = ModelWrapper(student_backbone, S_cls_head, None)
        self.teacher = ModelWrapper(teacher_backbone, T_cls_head, None)

        
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        # self.teacher.backbone.load_state_dict(self.student.backbone.state_dict())
        # self.teacher.cls_head.load_state_dict(self.student.cls_head.state_dict())
        # self.teacher.patch_head.proj_head.load_state_dict(self.student.patch_head.proj_head.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False


    def mask_patch_pair_embed(self, s_crops, t_crops, is_global=True):
        b, nc, w, h = s_crops.shape
        s_patches = self.student.backbone.patch_embed(s_crops)
        t_patches = self.teacher.backbone.patch_embed(t_crops)

        B, L, C = s_patches.shape

        if is_global:
            all_noise_sort = torch.argsort(torch.rand(B, L, device=s_patches.device), dim=1)
            s_l_num_patch = int(round(L * (1 - self.t_ratio)))
            s_num_patch = int(round(L * self.s_ratio))
            s_l_mask = all_noise_sort < s_l_num_patch
            s_mask = all_noise_sort < s_num_patch
            l_mask = s_l_mask & (~s_mask)
            t_mask = ~s_l_mask

            s_l_noise_sort = all_noise_sort[s_l_mask].view((all_noise_sort.shape[0], -1))
            s_mask_ofsl = s_l_noise_sort < s_num_patch
            cs_mask_ofcsl = torch.cat((torch.ones(B, 1, device=s_patches.device).bool(), s_mask_ofsl), dim=1)

            s_patches = torch.where(s_mask.unsqueeze(-1), s_patches,  self.s_mask_token)

            c_s_mask = torch.cat((torch.ones(B, 1, device=s_patches.device).bool(), s_mask), dim=1)
            c_t_mask = torch.cat((torch.ones(B, 1, device=s_patches.device).bool(), t_mask), dim=1)
            l_mask_ofcslt = torch.cat((torch.zeros(B, 1, device=s_patches.device).bool(), l_mask), dim=1)

            s_patches = torch.cat((self.student.backbone.cls_token.expand(s_patches.shape[0], -1, -1), s_patches), dim=1)
            t_patches = torch.cat((self.teacher.backbone.cls_token.expand(t_patches.shape[0], -1, -1), t_patches), dim=1)
            s_patches = s_patches + self.student.backbone.interpolate_pos_encoding(s_patches, w, h)
            t_patches = t_patches + self.teacher.backbone.interpolate_pos_encoding(t_patches, w, h)

            l_patches = s_patches[l_mask_ofcslt].view((s_patches.shape[0], -1, s_patches.shape[-1]))
            s_patches = s_patches[c_s_mask].view((s_patches.shape[0], -1, s_patches.shape[-1]))

            t_patches = t_patches[c_t_mask].view((t_patches.shape[0], -1, t_patches.shape[-1]))

            s_global_target = self.patchify(s_crops)
            s_global_target = s_global_target[l_mask].view(    
                (s_global_target.shape[0], -1, s_global_target.shape[-1]))  # remove patches that not need to learn

            return s_patches, t_patches, l_patches, s_global_target, cs_mask_ofcsl
        else:
            all_noise_sort = torch.argsort(torch.rand(B, L, device=s_patches.device), dim=1)
            s_l_num_patch = int(round(L * (1 - self.t_ratio)))
            s_num_patch = int(round(L * self.s_ratio))
            s_mask = all_noise_sort < s_num_patch
            t_mask = all_noise_sort >= s_l_num_patch

            c_s_mask = torch.cat((torch.ones(B, 1, device=s_patches.device).bool(), s_mask), dim=1)
            c_t_mask = torch.cat((torch.ones(B, 1, device=s_patches.device).bool(), t_mask), dim=1)

            s_patches = torch.cat((self.student.backbone.cls_token.expand(s_patches.shape[0], -1, -1), s_patches), dim=1)
            t_patches = torch.cat((self.teacher.backbone.cls_token.expand(t_patches.shape[0], -1, -1), t_patches), dim=1)
            s_patches = s_patches + self.student.backbone.interpolate_pos_encoding(s_patches, w, h)
            t_patches = t_patches + self.teacher.backbone.interpolate_pos_encoding(t_patches, w, h)

            s_patches = s_patches[c_s_mask].view((s_patches.shape[0], -1, s_patches.shape[-1]))
            t_patches = t_patches[c_t_mask].view((t_patches.shape[0], -1, t_patches.shape[-1]))

            return s_patches, t_patches, None, None, None


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.cfg.NET.PATCH_SIZE
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


    # teacher output
    @torch.no_grad()
    def teacher_nogard_forward(self, t_global_crops, t_local_crops):
        x = [t_global_crops, t_local_crops]
        for blk in self.teacher.backbone.blocks:
            x = blk(x)
        t_global_backbone_output = self.teacher.backbone.norm(x[0])
        t_local_backbone_output = self.teacher.backbone.norm(x[1])
        
        t_global_cls_tokens = t_global_backbone_output[:, 0]
        t_local_cls_tokens = t_local_backbone_output[:, 0]
        # t_global_patch_tokens = t_global_backbone_output[:, 1:]
        # t_local_patch_tokens = t_local_backbone_output[:, 1:]
        
        t_global_cls_tokens_after_head = self.teacher.cls_head(t_global_cls_tokens)
        t_local_cls_tokens_after_head = self.teacher.cls_head(t_local_cls_tokens)

        # t_global_patch_tokens_after_head = self.teacher.patch_head(t_global_patch_tokens)
        # t_local_patch_tokens_after_head = self.teacher.patch_head(t_local_patch_tokens)
      
        # return t_global_cls_tokens_after_head, t_global_patch_tokens_after_head, t_local_cls_tokens_after_head, None
        return t_global_cls_tokens_after_head, None, t_local_cls_tokens_after_head, None

    def forward(self, images):
        s_global_crops = images["S_global_crops"]
        s_local_crops = images["S_local_crops"]
        t_global_crops = images["T_global_crops"]
        t_local_crops = images["T_local_crops"]

        s_global_input, t_global_input, l_patches, s_global_target, cs_mask_ofcsl = self.mask_patch_pair_embed(s_global_crops, t_global_crops, is_global=True)
        s_local_input, t_local_input, _, _, _ = self.mask_patch_pair_embed(s_local_crops, t_local_crops, is_global=False)

        t_global_cls_tokens_after_head, t_global_patch_tokens_after_head, t_local_cls_tokens_after_head, t_local_patch_tokens_after_head = self.teacher_nogard_forward(
            t_global_input, t_local_input)
        
        x = [s_global_input, s_local_input]
        for i, blk in enumerate(self.student.backbone.blocks):
            if i == self.cfg.NET.DEPTH // 2:
                s_global_input, s_local_input = x
                s_global_input_temp = torch.empty(cs_mask_ofcsl.shape[0], cs_mask_ofcsl.shape[1], 
                                                   s_global_input.shape[2], device=s_global_input.device)
                s_global_input_temp[cs_mask_ofcsl] = s_global_input.view(-1, s_global_input.shape[-1])
                s_global_input_temp[~cs_mask_ofcsl] = l_patches.view(-1, l_patches.shape[-1])
                x = [s_global_input_temp, s_local_input]
            x = blk(x)
        s_global_backbone_output = self.student.backbone.norm(x[0])
        s_local_backbone_output = self.student.backbone.norm(x[1])
        
        s_global_cls_tokens = s_global_backbone_output[:, 0]
        s_local_cls_tokens = s_local_backbone_output[:, 0]
        # s_l_global_patch_tokens = s_global_backbone_output[:, 1:]
        # s_local_patch_tokens = s_local_backbone_output[:, 1:]

        s_global_pred_tokens = s_global_backbone_output[~cs_mask_ofcsl].view(
            (s_global_backbone_output.shape[0], -1, s_global_backbone_output.shape[-1]))  # remove s part of patches
        # s_global_patch_tokens = s_l_global_patch_tokens[s_mask_ofsl].view(
        #     (s_l_global_patch_tokens.shape[0], -1, s_l_global_patch_tokens.shape[-1]))  # remove l part of patches
        
        s_global_cls_tokens_after_head = self.student.cls_head(s_global_cls_tokens)
        s_local_cls_tokens_after_head = self.student.cls_head(s_local_cls_tokens)

        # s_global_patch_tokens_after_head = self.student.patch_head(s_global_patch_tokens)
        # s_local_patch_tokens_after_head = self.student.patch_head(s_local_patch_tokens)

        s_global_pred_tokens_after_head = self.pred_head(s_global_pred_tokens)

        student_output = dict()
        student_output['local_cls_tokens'] = s_local_cls_tokens_after_head
        student_output['global_cls_tokens'] = s_global_cls_tokens_after_head
        # student_output['local_patch_tokens'] = s_local_patch_tokens_after_head
        # student_output['global_patch_tokens'] = s_global_patch_tokens_after_head
        student_output['global_pred_tokens'] = s_global_pred_tokens_after_head
        student_output['global_patch_target'] = s_global_target

        teacher_output = dict()
        teacher_output['local_cls_tokens'] = t_local_cls_tokens_after_head
        teacher_output['global_cls_tokens'] = t_global_cls_tokens_after_head
        # teacher_output['local_patch_tokens'] = t_local_patch_tokens_after_head
        # teacher_output['global_patch_tokens'] = t_global_patch_tokens_after_head

        return student_output, teacher_output