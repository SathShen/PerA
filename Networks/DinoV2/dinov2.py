import torch
import torch.nn.functional as F
from .layers import DINOHead
from .vision_transformer import DinoV2VisionTransformer
from torch import nn
from collections import defaultdict
from xformers.ops import fmha


# def has_batchnorms(model):
#     bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
#     for name, module in model.named_modules():
#         if isinstance(module, bn_types):
#             return True
#     return False


class ModelWrapper(nn.Module):
    def __init__(self, backbone, dino_head, ibot_head):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head


# new module class
class DinoV2Smuc(nn.Module):
    def __init__(self, cfg):
        super(DinoV2Smuc, self).__init__()
        self.cfg = cfg
        self.student_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO,
                                                      drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                      drop_path_uniform=True)
        self.teacher_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO)
        self.S_dino_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.T_dino_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.S_ibot_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.T_ibot_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)

        self.student = ModelWrapper(self.student_backbone, self.S_dino_head, self.S_ibot_head)
        self.teacher = ModelWrapper(self.teacher_backbone, self.T_dino_head, self.T_ibot_head)

        # if has_batchnorms(self.student):
        #     self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
        #     self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False


    # teacher output
    @torch.no_grad()
    def get_teacher_output(self, images):
        t_global_backbone_output_dict, t_local_backbone_output_dict  = self.teacher.backbone(
            [images["T_global_crops"], images["T_local_crops"]], masks=[None, None], is_training=True)
        
        t_global_cls_tokens = t_global_backbone_output_dict["x_norm_clstoken"]
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        # if self.cfg.AUG.NUM_GLOBAL != 1:
        #     t_global_cls_tokens = t_global_cls_tokens.chunk(self.cfg.AUG.NUM_GLOBAL)[::-1]
        #     t_global_cls_tokens = torch.cat(t_global_cls_tokens)

        t_local_cls_tokens = t_local_backbone_output_dict["x_norm_clstoken"]
        # if self.cfg.AUG.NUM_LOCAL != 1:
        #     t_local_cls_tokens = t_local_cls_tokens.chunk(self.cfg.AUG.NUM_LOCAL)[::-1]
        #     t_local_cls_tokens = torch.cat(t_local_cls_tokens)

        patch_tokens = t_global_backbone_output_dict["x_norm_patchtokens"]
        _dim = patch_tokens.shape[-1]
        num_cls_tokens = t_global_cls_tokens.shape[0]

        buffer_tensor_teacher = patch_tokens.new_zeros(images["upperbound"], _dim)
        torch.index_select(patch_tokens.flatten(0, 1), dim=0, index=images["mask_indices_list"], 
                           out=buffer_tensor_teacher[:images["num_masked"].item()])
        
        t_global_cls_tokens_after_head = self.teacher.dino_head(t_global_cls_tokens)
        t_local_cls_tokens_after_head = self.teacher.dino_head(t_local_cls_tokens)

        masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[:images["num_masked"].item()]
      
        return t_global_cls_tokens_after_head, masked_teacher_patch_tokens_after_head, t_local_cls_tokens_after_head

    def forward(self, images):
        # S_global_crop:storch.Size([128, 3, 448, 448]) S_local_crops:torch.Size([512, 3, 96, 96]) masks:torch.Size([128, 784]
        t_global_cls_tokens, t_global_patch_tokens, t_local_cls_tokens = self.get_teacher_output(images)   # tgc tgp

        s_global_backbone_output_dict, s_local_backbone_output_dict = self.student.backbone(
            [images['S_global_crops'], images['S_local_crops']], masks=[images['masks'], None], is_training=True)

        inputs_for_student_head_list = []
        _dim = s_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
        s_local_cls_tokens = s_local_backbone_output_dict["x_norm_clstoken"]
        s_global_cls_tokens = s_global_backbone_output_dict["x_norm_clstoken"]
        s_global_patch_tokens = s_global_backbone_output_dict["x_norm_patchtokens"]
        inputs_for_student_head_list.append(s_local_cls_tokens.unsqueeze(0))
        inputs_for_student_head_list.append(s_global_cls_tokens.unsqueeze(0))
        buffer_patch_tokens = s_global_patch_tokens.new_zeros(images["upperbound"], _dim)
        buffer_patch_tokens[:images['num_masked']].copy_(torch.index_select(s_global_patch_tokens.flatten(0, 1), 
                                                                            dim=0, index=images['mask_indices_list']))
        s_global_patch_tokens = self.student.ibot_head(buffer_patch_tokens)[:images['num_masked']]    # sgp

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        s_local_cls_tokens = outputs_list.pop(0).squeeze(0)    # slc
        s_global_cls_tokens = outputs_list.pop(0).squeeze(0)    # sgc

        student_output = dict()
        student_output['local_cls_tokens'] = s_local_cls_tokens
        student_output['global_cls_tokens'] = s_global_cls_tokens
        student_output['global_patch_tokens'] = s_global_patch_tokens

        teacher_output = dict()
        teacher_output['local_cls_tokens'] = t_local_cls_tokens
        teacher_output['global_cls_tokens'] = t_global_cls_tokens
        teacher_output['global_patch_tokens'] = t_global_patch_tokens

        # local_cls_tokens:torch.Size([512, 65536]) global_cls_tokens:torch.Size([128, 65536]) global_patch_tokens:torch.Size([15020, 65536])
        # s exacly same as t
        return student_output, teacher_output

    
    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.children():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups
    
    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = self.get_params_groups_with_decay(
            model=m,
            lr_decay_rate=0.9,   # layerwise_decay: 0.9
            patch_embed_lr_mult=0.2,   # patch_embed_lr_mult: 0.2
        )
        fused_params_groups = self.fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
    
    def get_params_groups_with_decay(self, model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
        chunked_blocks = False
        if hasattr(model, "n_blocks"):
            # logger.info("chunked fsdp")
            n_blocks = model.n_blocks
            chunked_blocks = model.chunked_blocks
        elif hasattr(model, "blocks"):
            # logger.info("first code branch")
            n_blocks = len(model.blocks)
        elif hasattr(model, "backbone"):
            # logger.info("second code branch")
            n_blocks = len(model.backbone.blocks)
        else:
            # logger.info("else code branch")
            n_blocks = 0
        all_param_groups = []

        for name, param in model.named_parameters():
            name = name.replace("_fsdp_wrapped_module.", "")
            if not param.requires_grad:
                continue
            decay_rate = self.get_vit_lr_decay_rate(
                name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
            )
            d = {"params": param, "is_last_layer": False, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

            if "last_layer" in name:
                d.update({"is_last_layer": True})

            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

            all_param_groups.append(d)
            # logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

        return all_param_groups
    
    def fuse_params_groups(self, all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
        fused_params_groups = defaultdict(lambda: {"params": []})
        for d in all_params_groups:
            identifier = ""
            for k in keys:
                identifier += k + str(d[k]) + "_"

            for k in keys:
                fused_params_groups[identifier][k] = d[k]
            fused_params_groups[identifier]["params"].append(d["params"])

        return fused_params_groups.values()
    
    def get_vit_lr_decay_rate(self ,name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
        """
        Calculate lr decay rate for different ViT blocks.
        Args:
            name (string): parameter name.
            lr_decay_rate (float): base lr decay rate.
            num_layers (int): number of ViT blocks.
        Returns:
            lr decay rate for the given parameter.
        """
        layer_id = num_layers + 1
        if name.startswith("backbone") or force_is_backbone:
            if ".pos_embed" in name or ".patch_embed" in name or ".mask_token" in name or ".cls_token" in name:
                layer_id = 0
            elif force_is_backbone and (
                "pos_embed" in name or "patch_embed" in name or "mask_token" in name or "cls_token" in name
            ):
                layer_id = 0
            elif ".blocks." in name and ".residual." not in name:
                layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
            elif chunked_blocks and "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
            elif "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

        return lr_decay_rate ** (num_layers + 1 - layer_id)
    


class DinoV2(nn.Module):
    def __init__(self, cfg):
        super(DinoV2, self).__init__()
        self.student_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO,
                                                      drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                      drop_path_uniform=True)
        self.teacher_backbone = DinoV2VisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO)
        self.S_dino_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.T_dino_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.S_ibot_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)
        self.T_ibot_head = DINOHead(in_dim=cfg.NET.EMBED_DIM, 
                                  out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                  use_bn=False, 
                                  nlayers=cfg.NET.DINO.NUM_HEAD_LAYERS, 
                                  hidden_dim=cfg.NET.DINO.HEAD_HIDDEN_DIM, 
                                  bottleneck_dim=cfg.NET.DINO.HEAD_BOTTLENECK_DIM)

        self.student = ModelWrapper(self.student_backbone, self.S_dino_head, self.S_ibot_head)
        self.teacher = ModelWrapper(self.teacher_backbone, self.T_dino_head, self.T_ibot_head)

        # if has_batchnorms(self.student):
        #     self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
        #     self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        for p in self.teacher.parameters():
            p.requires_grad = False


    # teacher output
    @torch.no_grad()
    def get_teacher_output(self, images):
        T_backbone_output_dict = self.teacher.backbone(images["global_crops"], is_training=True)  # group t output
        cls_tokens = T_backbone_output_dict["x_norm_clstoken"].chunk(2)
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        cls_tokens = torch.cat((cls_tokens[1], cls_tokens[0]))

        patch_tokens = T_backbone_output_dict["x_norm_patchtokens"]
        _dim = patch_tokens.shape[-1]
        num_cls_tokens = cls_tokens.shape[0]

        buffer_tensor_teacher = patch_tokens.new_zeros(images["upperbound"], _dim)

        torch.index_select(patch_tokens.flatten(0, 1), dim=0, index=images["mask_indices_list"], 
                           out=buffer_tensor_teacher[:images["num_masked"].item()])
        teacher_cls_tokens_after_head = self.teacher.dino_head(cls_tokens)
        masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[:images["num_masked"].item()]
      
        return teacher_cls_tokens_after_head, masked_teacher_patch_tokens_after_head

    def forward(self, images):
        t_global_cls_tokens, t_global_patch_tokens = self.get_teacher_output(images)   # tgc tgp

        s_global_backbone_output_dict, s_local_backbone_output_dict = self.student.backbone(
            [images['global_crops'], images['local_crops']], masks=[images['masks'], None], is_training=True)

        inputs_for_student_head_list = []
        _dim = s_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
        s_local_cls_tokens = s_local_backbone_output_dict["x_norm_clstoken"]
        s_global_cls_tokens = s_global_backbone_output_dict["x_norm_clstoken"]
        s_global_patch_tokens = s_global_backbone_output_dict["x_norm_patchtokens"]
        inputs_for_student_head_list.append(s_local_cls_tokens.unsqueeze(0))
        inputs_for_student_head_list.append(s_global_cls_tokens.unsqueeze(0))
        buffer_patch_tokens = s_global_patch_tokens.new_zeros(images["upperbound"], _dim)
        buffer_patch_tokens[:images['num_masked']].copy_(torch.index_select(s_global_patch_tokens.flatten(0, 1), 
                                                                            dim=0, index=images['mask_indices_list']))
        s_global_patch_tokens = self.student.ibot_head(buffer_patch_tokens)[:images['num_masked']]    # sgp

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))
        s_local_cls_tokens = outputs_list.pop(0).squeeze(0)    # slc
        s_global_cls_tokens = outputs_list.pop(0).squeeze(0)    # sgc

        student_output = dict()
        student_output['local_cls_tokens'] = s_local_cls_tokens
        student_output['global_cls_tokens'] = s_global_cls_tokens
        student_output['global_patch_tokens'] = s_global_patch_tokens

        teacher_output = dict()
        teacher_output['global_cls_tokens'] = t_global_cls_tokens
        teacher_output['global_patch_tokens'] = t_global_patch_tokens

        return student_output, teacher_output

    
    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.children():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups
    
    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = self.get_params_groups_with_decay(
            model=m,
            lr_decay_rate=0.9,   # layerwise_decay: 0.9
            patch_embed_lr_mult=0.2,   # patch_embed_lr_mult: 0.2
        )
        fused_params_groups = self.fuse_params_groups(params_groups)
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups
    
    def get_params_groups_with_decay(self, model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
        chunked_blocks = False
        if hasattr(model, "n_blocks"):
            # logger.info("chunked fsdp")
            n_blocks = model.n_blocks
            chunked_blocks = model.chunked_blocks
        elif hasattr(model, "blocks"):
            # logger.info("first code branch")
            n_blocks = len(model.blocks)
        elif hasattr(model, "backbone"):
            # logger.info("second code branch")
            n_blocks = len(model.backbone.blocks)
        else:
            # logger.info("else code branch")
            n_blocks = 0
        all_param_groups = []

        for name, param in model.named_parameters():
            name = name.replace("_fsdp_wrapped_module.", "")
            if not param.requires_grad:
                continue
            decay_rate = self.get_vit_lr_decay_rate(
                name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
            )
            d = {"params": param, "is_last_layer": False, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

            if "last_layer" in name:
                d.update({"is_last_layer": True})

            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                d.update({"wd_multiplier": 0.0})

            if "patch_embed" in name:
                d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

            all_param_groups.append(d)
            # logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}""")

        return all_param_groups
    
    def fuse_params_groups(self, all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
        fused_params_groups = defaultdict(lambda: {"params": []})
        for d in all_params_groups:
            identifier = ""
            for k in keys:
                identifier += k + str(d[k]) + "_"

            for k in keys:
                fused_params_groups[identifier][k] = d[k]
            fused_params_groups[identifier]["params"].append(d["params"])

        return fused_params_groups.values()
    
    def get_vit_lr_decay_rate(self ,name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
        """
        Calculate lr decay rate for different ViT blocks.
        Args:
            name (string): parameter name.
            lr_decay_rate (float): base lr decay rate.
            num_layers (int): number of ViT blocks.
        Returns:
            lr decay rate for the given parameter.
        """
        layer_id = num_layers + 1
        if name.startswith("backbone") or force_is_backbone:
            if ".pos_embed" in name or ".patch_embed" in name or ".mask_token" in name or ".cls_token" in name:
                layer_id = 0
            elif force_is_backbone and (
                "pos_embed" in name or "patch_embed" in name or "mask_token" in name or "cls_token" in name
            ):
                layer_id = 0
            elif ".blocks." in name and ".residual." not in name:
                layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
            elif chunked_blocks and "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
            elif "blocks." in name and "residual." not in name:
                layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

        return lr_decay_rate ** (num_layers + 1 - layer_id)
