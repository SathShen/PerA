import sys
sys.path.append('./')
import torch
from Networks import DinoV1, DinoV2, PerA, MoCoV3, MAE
from Networks.DinoV1.vision_transformer import VisionTransformer as DinoV1_VIT
from Networks.DinoV2.vision_transformer import DinoV2VisionTransformer as DinoV2_VIT
from Networks.MoCoV3.vit_moco import VisionTransformerMoCo
from Networks.MAE.mae_vit import MAEVisionTransformer

from Utils.loss import *
import numpy as np
from Utils.dataset import *
from Utils.augmentation import *
# import deepspeed
import time
import logging
import functools
from termcolor import colored



class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

class CosineScheduler(object):
    def __init__(self, start_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0, is_restart=False, T_0=5, T_mult=2):
        super().__init__()
        """
        freeze -> warmup -> cosine
        Cosine scheduler with warmup and restarts.
        Args:
            start_value (float): start cosine value of the scheduler
            final_value (float): final value of the scheduler
            total_iters (int): total number of iterations
            warmup_iters (int): number of warmup iterations
            start_warmup_value (float): initial value of the warmup
            freeze_iters (int): number of iterations to freeze the scheduler
            is_restart (bool): whether to restart the scheduler at the end of each cycle
            T_0 (int): number of iterations in the first cycle
            T_mult (int): factor to increase T_0 at each restart
        """
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value, start_value, warmup_iters)
        
        if is_restart:
            if T_mult <= 1 or T_0 <= 0:
                raise ValueError("T_0 must be positive and T_mult must be greater than 1 for restarts.")
            num_iters = total_iters - warmup_iters - freeze_iters
            iters, T_is = self.get_restart_iters(num_iters, T_0, T_mult)
            schedule = final_value + 0.5 * (start_value - final_value) * (1 + np.cos(np.pi * iters / T_is))
        else:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + 0.5 * (start_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
        
    def get_restart_iters(self, num_iters, T_0, T_mult):
        iter_accu = 0
        num_periods = 0
        sub_iters_tuple = ()
        T_i_tuple = () 
        while iter_accu < num_iters:
            T_i = T_mult ** num_periods * T_0      # 2 ^ 0 * 5 = 5, 2 ^ 1 * 5 = 10
            iter_accu += T_i

            num_T = T_i
            if iter_accu > num_iters:
                num_T = T_i - (iter_accu - num_iters)
            sub_T_i = np.arange(num_T)
            sub_T_i.fill(T_i)
            T_i_tuple += (sub_T_i,)          # [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, ...]
            
            sub_iter = np.arange(num_T)
            sub_iters_tuple += (sub_iter,)
            num_periods += 1
        iters = np.concatenate(sub_iters_tuple)
        T_is = np.concatenate(T_i_tuple)
        return iters, T_is


@functools.lru_cache()
def build_logger(output_dir, name='', cfg_note='default', rank=0):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(f"{output_dir}/{name}_{cfg_note}_{time.strftime('%y%m%d%H%M%S')}.log", mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger


def build_dataset(cfg, data_path, mode='Pretrain', is_aug=True):
    trans = build_transform(cfg, mode=mode, is_aug=is_aug)
    
    if mode == 'Pretrain':
        dataset = PretrainDataset(cfg, trans, data_path)
    else:
        raise NotImplementedError('Not implemented mode: {}'.format(mode))
    return dataset



def build_net(cfg, is_pretrain=True):
    if is_pretrain == True:
        if cfg.NET.NAME == 'dinov1' or cfg.NET.NAME == 'dino':
            net = DinoV1(cfg)
        elif cfg.NET.NAME == 'dinov2':
            net = DinoV2(cfg)
        elif cfg.NET.NAME == 'pera':
            net = PerA(cfg)
        elif cfg.NET.NAME == 'mocov3':
            net = MoCoV3(cfg)
        elif cfg.NET.NAME == 'mae':
            net = MAE(cfg)
        else:
            raise NotImplementedError(f"Unknown net: {cfg.NET.NAME}")
        return net
    else:
        if cfg.DISTILL.IS_DISTILL:
            if cfg.FINETUNE.IS_FINETUNE or cfg.IS_EVAL:
                if cfg.DISTILL.STUDENT_ARCH == 'vitt':
                    embed_dim = 192
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = DinoV2_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                        patch_size=cfg.NET.PATCH_SIZE, 
                                        embed_dim=embed_dim,
                                        depth=12,
                                        num_heads=3,
                                        mlp_ratio=4,
                                        drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        s = 12 // cfg.NET.ADAPTER.NUM_POINTS
                        interaction_indexes = [[x, x + s - 1] for x in range(0, d, s)]
                        backbone = DinoV2_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=embed_dim,
                                                    depth=12,
                                                    num_heads=3,
                                                    mlp_ratio=4,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                    drop_path_uniform=True,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=interaction_indexes,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                elif cfg.DISTILL.STUDENT_ARCH == 'vits':
                    embed_dim = 384
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = DinoV2_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                        patch_size=cfg.NET.PATCH_SIZE, 
                                        embed_dim=embed_dim,
                                        depth=12,
                                        num_heads=6,
                                        mlp_ratio=4,
                                        drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        s = 12 // cfg.NET.ADAPTER.NUM_POINTS
                        interaction_indexes = [[x, x + s - 1] for x in range(0, d, s)]
                        backbone = DinoV2_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=embed_dim,
                                                    depth=12,
                                                    num_heads=6,
                                                    mlp_ratio=4,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                    drop_path_uniform=True,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=interaction_indexes,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                elif cfg.DISTILL.STUDENT_ARCH == 'vitb':
                    embed_dim = 768
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = DinoV2_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                        patch_size=cfg.NET.PATCH_SIZE, 
                                        embed_dim=embed_dim,
                                        depth=12,
                                        num_heads=12,
                                        mlp_ratio=4,
                                        drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        d = 12
                        s = 12 // cfg.NET.ADAPTER.NUM_POINTS
                        interaction_indexes = [[x, x + s - 1] for x in range(0, d, s)]
                        backbone = DinoV2_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=embed_dim,
                                                    depth=12,
                                                    num_heads=12,
                                                    mlp_ratio=4,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                    drop_path_uniform=True,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=interaction_indexes,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                elif cfg.DISTILL.STUDENT_ARCH == 'vitl':
                    embed_dim = 1024
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = DinoV2_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                        patch_size=cfg.NET.PATCH_SIZE, 
                                        embed_dim=embed_dim,
                                        depth=24,
                                        num_heads=16,
                                        mlp_ratio=4,
                                        drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        s = 24 // cfg.NET.ADAPTER.NUM_POINTS
                        interaction_indexes = [[x, x + s - 1] for x in range(0, d, s)]
                        backbone = DinoV2_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=embed_dim,
                                                    depth=24,
                                                    num_heads=16,
                                                    mlp_ratio=4,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                    drop_path_uniform=True,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=interaction_indexes,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                else:
                    raise NotImplementedError(f"Unknown student_arch: {cfg.DISTILL.STUDENT_ARCH}")
            else:
                net = DistillNet(cfg)
                return net
        else:
            embed_dim = cfg.NET.EMBED_DIM
            if cfg.FINETUNE.BACKBONE == 'swin':
                pass
            elif cfg.FINETUNE.BACKBONE =='swinv2':
                pass
            elif cfg.FINETUNE.BACKBONE == 'vit':
                if cfg.NET.NAME == 'dinov1' or cfg.NET.NAME == 'dino':
                    backbone = DinoV1_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                            patch_size=cfg.NET.PATCH_SIZE,
                                            embed_dim=cfg.NET.EMBED_DIM,
                                            depth=cfg.NET.DEPTH,
                                            num_heads=cfg.NET.NUM_HEADS,
                                            mlp_ratio=cfg.NET.MLP_RATIO,
                                            drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                            )
                elif cfg.NET.NAME == 'dinov2' or cfg.NET.NAME == 'dinov2smuc' or cfg.NET.NAME == 'pera':
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = DinoV2_VIT(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                patch_size=cfg.NET.PATCH_SIZE, 
                                                embed_dim=cfg.NET.EMBED_DIM,
                                                depth=cfg.NET.DEPTH,
                                                num_heads=cfg.NET.NUM_HEADS,
                                                mlp_ratio=cfg.NET.MLP_RATIO,
                                                drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                drop_path_uniform=True)
                    else:
                        backbone = DinoV2_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=cfg.NET.EMBED_DIM,
                                                    depth=cfg.NET.DEPTH,
                                                    num_heads=cfg.NET.NUM_HEADS,
                                                    mlp_ratio=cfg.NET.MLP_RATIO,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE,
                                                    drop_path_uniform=True,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=cfg.NET.ADAPTER.INTERACTION_INDEXES,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)                
                elif cfg.NET.NAME =='mocov3':
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = VisionTransformerMoCo(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=cfg.NET.EMBED_DIM,
                                                    depth=cfg.NET.DEPTH,
                                                    num_heads=cfg.NET.NUM_HEADS,
                                                    mlp_ratio=cfg.NET.MLP_RATIO,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        backbone = MoCoV3_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=cfg.NET.EMBED_DIM,
                                                    depth=cfg.NET.DEPTH,
                                                    num_heads=cfg.NET.NUM_HEADS,
                                                    mlp_ratio=cfg.NET.MLP_RATIO,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=cfg.NET.ADAPTER.INTERACTION_INDEXES,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                elif cfg.NET.NAME =='mae':
                    if cfg.FINETUNE.TYPE == 'ic'and not cfg.FINETUNE.IC.IS_PATCH_INPUT:
                        backbone = MAEVisionTransformer(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=cfg.NET.EMBED_DIM,
                                                    depth=cfg.NET.DEPTH,
                                                    num_heads=cfg.NET.NUM_HEADS,
                                                    mlp_ratio=cfg.NET.MLP_RATIO,
                                                    drop_path_rate=cfg.NET.DROP_PATH_RATE)
                    else:
                        backbone = MAE_ViTAdapter(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                    patch_size=cfg.NET.PATCH_SIZE, 
                                                    embed_dim=cfg.NET.EMBED_DIM,
                                                    depth=cfg.NET.DEPTH,
                                                    num_heads=cfg.NET.NUM_HEADS,
                                                    mlp_ratio=cfg.NET.MLP_RATIO,
                                                    conv_inplane=cfg.NET.ADAPTER.CONV_INPLANE,
                                                    n_points=cfg.NET.ADAPTER.NUM_POINTS,
                                                    deform_num_heads=cfg.NET.ADAPTER.DEFORM_NUM_HEADS,
                                                    init_values=cfg.NET.ADAPTER.INIT_VALUES,
                                                    interaction_indexes=cfg.NET.ADAPTER.INTERACTION_INDEXES,
                                                    with_cffn=cfg.NET.ADAPTER.IS_WITH_CFFN,
                                                    cffn_ratio=cfg.NET.ADAPTER.CFFN_RATIO,
                                                    add_vit_feature=cfg.NET.ADAPTER.IS_ADD_VIT_FEATURE,
                                                    use_extra_extractor=cfg.NET.ADAPTER.IS_USE_EXTRA_EXTRACTOR,
                                                    with_cp=cfg.NET.ADAPTER.IS_USE_CHECKPOINT)
                else:
                    raise NotImplementedError(f"Unknown net: {cfg.NET.NAME}")      
            else:
                raise NotImplementedError(f"Unknown backbone: {cfg.FINETUNE.BACKBONE}")
        
        # build head
        if cfg.FINETUNE.TYPE == 'ic':
            if cfg.FINETUNE.IC.HEAD == 'mlp':
                head = MLPHead(embed_dim=embed_dim, 
                                hidden_channels=cfg.FINETUNE.IC.HIDDEN_CHANNELS, 
                                num_classes=cfg.FINETUNE.IC.NUM_CLASSES, 
                                num_layers=cfg.FINETUNE.IC.NUM_LAYERS)
            else:
                raise NotImplementedError(f"Unknown head: {cfg.FINETUNE.IC.HEAD}")      
        elif cfg.FINETUNE.TYPE == 'seg':
            if cfg.FINETUNE.SEG.HEAD == 'fcn':
                pass
            elif cfg.FINETUNE.SEG.HEAD == 'unet':
                pass
            elif cfg.FINETUNE.SEG.HEAD == 'uper' or cfg.FINETUNE.SEG.HEAD == 'upernet':
                head = UPerNetHead(embed_dim=embed_dim, 
                                num_classes=cfg.FINETUNE.SEG.NUM_CLASSES,
                                output_size=cfg.DATA.INPUT_SIZE,
                                hidden_dim=cfg.FINETUNE.SEG.HIDDEN_CHANNELS,
                                out_dim=cfg.FINETUNE.SEG.OUT_CHANNELS)
            else:
                raise NotImplementedError(f"Unknown head: {cfg.FINETUNE.SEG.HEAD}")            
        elif cfg.FINETUNE.TYPE == 'cd':
            if cfg.FINETUNE.CD.HEAD == 'unet':
                head = CDSUNetHead(embed_dim=embed_dim, 
                                    num_classes=cfg.FINETUNE.CD.NUM_CLASSES)
            elif cfg.FINETUNE.CD.HEAD == 'uper' or cfg.FINETUNE.CD.HEAD == 'upernet':
                head = CDUPerNetHead(embed_dim=embed_dim, 
                                num_classes=cfg.FINETUNE.CD.NUM_CLASSES,
                                output_size=cfg.DATA.INPUT_SIZE,
                                hidden_dim=cfg.FINETUNE.CD.HIDDEN_CHANNELS,
                                out_dim=cfg.FINETUNE.CD.OUT_CHANNELS)
            elif cfg.FINETUNE.CD.HEAD == 'bit':
                head = BITHead(embed_dim=embed_dim,
                            output_nc=cfg.FINETUNE.CD.NUM_CLASSES)
            else:
                raise NotImplementedError(f"Unknown head: {cfg.FINETUNE.CD.HEAD}")            
        elif cfg.FINETUNE.TYPE == 'det':
            if cfg.FINETUNE.DET.HEAD == 'faster_rcnn' or cfg.FINETUNE.DET.HEAD == 'frcnn':
                head = FasterRCNNHead(embed_dim=embed_dim,
                                    num_classes=cfg.FINETUNE.DET.NUM_CLASSES, 
                                    input_size=cfg.DATA.INPUT_SIZE,
                                    patch_size=cfg.NET.PATCH_SIZE,
                                    hidden_dim=cfg.FINETUNE.DET.HIDDEN_CHANNELS,
                                    roi_size=cfg.FINETUNE.DET.ROI_SIZE)
            elif cfg.FINETUNE.DET.HEAD == 'off_frcnn':
                backbone.out_channels = backbone.embed_dim
                backbone.mask_token = None
                # for param in backbone.parameters():
                #     param.requires_grad = False
                net = FasterRCNN(backbone, num_classes=cfg.FINETUNE.DET.NUM_CLASSES + 1, min_size=cfg.DATA.INPUT_SIZE, max_size=cfg.DATA.INPUT_SIZE,
                                 image_mean=cfg.AUG.NORMALIZE.MEAN, image_std=cfg.AUG.NORMALIZE.STD,
                                 rpn_nms_thresh=0.8,
                                 box_score_thresh=0.01,
                                 box_detections_per_img=200,
                                 )
                return net
            else:
                raise NotImplementedError(f"Unknown head: {cfg.FINETUNE.DET.HEAD}")
        else:
            raise NotImplementedError(f"Unknown finetune type: {cfg.FINETUNE.TYPE}")

        net = FinetuneWrapper(cfg, backbone, head)
    return net


def build_loss(cfg, is_pretrain=True):
    if is_pretrain == True:
        if cfg.LOSS.NAME == 'dinov2smuc':
            loss_func = DinoV2SmucLoss(num_global_crops=cfg.AUG.NUM_GLOBAL,
                                num_local_crops=cfg.AUG.NUM_LOCAL, 
                                out_dim=cfg.NET.DINO.HEAD_OUT_DIM, 
                                centering=cfg.NET.DINO.CENTERING)
        elif cfg.LOSS.NAME == 'dinov1' or cfg.LOSS.NAME == 'dino':
            loss_func = DinoV1Loss(out_dim=cfg.NET.DINO.HEAD_OUT_DIM,
                                ncrops=cfg.AUG.NUM_LOCAL + cfg.AUG.NUM_GLOBAL,
                                student_temp=cfg.STUDENT_TEMP)
        elif cfg.LOSS.NAME == 'dinov2':
            loss_func = DinoV2Loss(num_global_crops=cfg.AUG.NUM_GLOBAL,
                                num_local_crops=cfg.AUG.NUM_LOCAL, 
                                out_dim=cfg.NET.DINO.HEAD_OUT_DIM, 
                                centering=cfg.NET.DINO.CENTERING)
        elif cfg.LOSS.NAME == 'pera':
            loss_func = PerALoss(num_global_crops=cfg.AUG.NUM_GLOBAL,
                                num_local_crops=cfg.AUG.NUM_LOCAL,
                                out_dim=cfg.NET.DINO.HEAD_OUT_DIM, 
                                centering=cfg.NET.DINO.CENTERING,
                                dino_loss_weight=cfg.NET.PERA.DINO_LOSS_WEIGHT,
                                # ibot_loss_weight=cfg.NET.PERA.IBOT_LOSS_WEIGHT,
                                koleo_loss_weight=cfg.NET.PERA.KOLEO_LOSS_WEIGHT,
                                mae_loss_weight=cfg.NET.PERA.MAE_LOSS_WEIGHT)
        elif cfg.LOSS.NAME =='mocov3':
            loss_func = MoCoV3Loss()
        elif cfg.LOSS.NAME =='mae':
            loss_func = MAELoss(patch_size=cfg.NET.PATCH_SIZE)
        else:
            raise NotImplementedError(f"Unknown loss: {cfg.LOSS.NAME}")
    else:
        if cfg.DISTILL.IS_DISTILL and not cfg.FINETUNE.IS_FINETUNE:
            loss_func = DistillLoss(num_global_crops=cfg.AUG.NUM_GLOBAL,
                                num_local_crops=cfg.AUG.NUM_LOCAL,
                                out_dim=cfg.NET.DINO.HEAD_OUT_DIM, 
                                centering=cfg.NET.DINO.CENTERING,
                                dino_loss_weight=cfg.NET.PERA.DINO_LOSS_WEIGHT,
                                # ibot_loss_weight=cfg.NET.PERA.IBOT_LOSS_WEIGHT,
                                koleo_loss_weight=cfg.NET.PERA.KOLEO_LOSS_WEIGHT,
                                # mae_loss_weight=cfg.NET.PERA.MAE_LOSS_WEIGHT
                                )
        else:
            if cfg.FINETUNE.TYPE == 'ic':
                if cfg.FINETUNE.IC.LOSS == 'ce':
                    loss_func = nn.CrossEntropyLoss()
                else:
                    raise NotImplementedError(f"Unknown loss: {cfg.FINETUNE.IC.LOSS}")
            elif cfg.FINETUNE.TYPE == 'seg':
                if cfg.FINETUNE.SEG.LOSS == 'ce':
                    loss_func = CrossEntropyLoss()
                elif cfg.FINETUNE.SEG.LOSS == 'bce':
                    loss_func = BCELoss()
                elif cfg.FINETUNE.SEG.LOSS == 'focal':
                    loss_func = FocalLoss()
                elif cfg.FINETUNE.SEG.LOSS == 'dice':
                    loss_func = DiceLoss()
                elif cfg.FINETUNE.SEG.LOSS == 'dicebce':
                    loss_func = DiceBCELoss()
                elif cfg.FINETUNE.SEG.LOSS == 'softiou':
                    loss_func = SoftIoULoss()
                else:
                    raise NotImplementedError(f"Unknown loss: {cfg.FINETUNE.SEG.LOSS}")
            elif cfg.FINETUNE.TYPE == 'det':
                if (cfg.FINETUNE.DET.HEAD in ['faster_rcnn', 'frcnn', 'fasterrcnn']) or (cfg.FINETUNE.DET.LOSS in ['faster_rcnn', 'frcnn', 'fasterrcnn', 'id']):
                    loss_func = IdentityLoss()
                else:
                    raise NotImplementedError(f"Unknown loss: {cfg.FINETUNE.DET.LOSS}")
            elif cfg.FINETUNE.TYPE == 'cd':
                if cfg.FINETUNE.CD.NUM_CLASSES == 2:
                    ig_idx = -100
                else:
                    ig_idx = cfg.FINETUNE.CD.MASK_ID

                if cfg.FINETUNE.CD.LOSS == 'ce':
                    loss_func = CDCrossEntropyLoss(num_classes=cfg.FINETUNE.CD.NUM_CLASSES, ignore_index=ig_idx)
                elif cfg.FINETUNE.CD.LOSS == 'focal':
                    loss_func = CDFocalLoss(num_classes=cfg.FINETUNE.CD.NUM_CLASSES, ignore_index=ig_idx)   
                elif cfg.FINETUNE.CD.LOSS == 'bce':
                    loss_func = CDBCELoss(num_classes=cfg.FINETUNE.CD.NUM_CLASSES, ignore_index=ig_idx)
            else:
                raise NotImplementedError(f"Unknown finetune type: {cfg.FINETUNE.TYPE}")
    return loss_func


def build_transform(cfg, mode='Pretrain', is_aug=True):
    if mode == 'Pretrain':
        if is_aug == False:
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])
        else:
            if cfg.NET.NAME == 'dinov2smuc':
                trans = DinoV2AugmentationSMuC(cfg)
            elif cfg.NET.NAME == 'dinov1' or cfg.NET.NAME == 'dino':
                trans = DinoV1Augmentation(cfg)
            elif cfg.NET.NAME == 'dinov2':
                trans = DinoV2Augmentation(cfg)
            elif cfg.NET.NAME == 'pera':
                trans = PerAAugmentation(cfg)
            elif cfg.NET.NAME =='mocov3':
                trans = MocoV3Augmentation(cfg)
            elif cfg.NET.NAME == 'mae':
                trans = transforms.Compose([
                        transforms.RandomResizedCrop(cfg.DATA.INPUT_SIZE, scale=cfg.AUG.CROP_SCALE, interpolation=3),  # 3 is bicubic
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
            else:
                raise NotImplementedError(f"Unknown net: {cfg.NET.NAME}")
    elif mode == 'KNN':
        if is_aug == False:
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])
        else:
            trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomResizedCrop(size=cfg.DATA.INPUT_SIZE,
                                                            scale=cfg.AUG.CROP_SCALE,
                                                            ratio=cfg.AUG.CROP_RATIO,
                                                            antialias=True),
                                        transforms.ColorJitter(brightness=cfg.AUG.INTENSITY,
                                                        contrast=cfg.AUG.CONTRAST,
                                                        saturation=cfg.AUG.SATURATION,
                                                        hue=cfg.AUG.HUE),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])
    elif mode == 'Finetune' or mode == 'Evaluate':
        if cfg.FINETUNE.TYPE == 'ic':
            if is_aug:
                trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomResizedCrop(size=cfg.DATA.INPUT_SIZE,
                                                                scale=cfg.AUG.CROP_SCALE,
                                                                ratio=cfg.AUG.CROP_RATIO,
                                                                antialias=True),
                                            transforms.ColorJitter(brightness=cfg.AUG.INTENSITY,
                                                            contrast=cfg.AUG.CONTRAST,
                                                            saturation=cfg.AUG.SATURATION,
                                                            hue=cfg.AUG.HUE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])
            else:
                trans = transforms.Compose([transforms.Resize(size=cfg.DATA.INPUT_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])

        elif cfg.FINETUNE.TYPE == 'seg':
            if is_aug:
                trans = SEGCompose([SEGRandomHorizontalFlip(),
                                    SEGRandomVerticalFlip(),
                                    SEGRandomResizedCrop(size=cfg.DATA.INPUT_SIZE, 
                                                        scale=cfg.AUG.CROP_SCALE, 
                                                        ratio=cfg.AUG.CROP_RATIO),
                                    SEGColorJitter(brightness=cfg.AUG.INTENSITY, 
                                                    contrast=cfg.AUG.CONTRAST, 
                                                    saturation=cfg.AUG.SATURATION, 
                                                    hue=cfg.AUG.HUE),
                                    SEGToTensor(),
                                    SEGNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
            else:
                trans = SEGCompose([SEGToTensor(),
                                    SEGNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])


        elif cfg.FINETUNE.TYPE == 'cd':
            if cfg.FINETUNE.CD.NUM_CLASSES == 2:
                if is_aug:
                    trans = CDSCompose([
                        CDSRandomHorizontalFlip(),
                        CDSRandomVerticalFlip(),
                        CDSRandomResizedCrop(size=cfg.DATA.INPUT_SIZE, 
                                            scale=cfg.AUG.CROP_SCALE, 
                                            ratio=cfg.AUG.CROP_RATIO),
                        CDSColorJitter(brightness=cfg.AUG.INTENSITY, 
                                        contrast=cfg.AUG.CONTRAST, 
                                        saturation=cfg.AUG.SATURATION, 
                                        hue=cfg.AUG.HUE),
                        CDSChangeOrder(0.1),
                        CDSToTensor(),
                        CDSNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
                else:
                    trans = CDSCompose([CDSToTensor(),
                                        CDSNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
            else:
                if is_aug:
                    trans = CDMCompose([CDMRandomHorizontalFlip(),
                                        CDMRandomVerticalFlip(),
                                        CDMRandomResizedCrop(size=cfg.DATA.INPUT_SIZE, 
                                                            scale=cfg.AUG.CROP_SCALE, 
                                                            ratio=cfg.AUG.CROP_RATIO),
                                        CDMColorJitter(brightness=cfg.AUG.INTENSITY, 
                                                        contrast=cfg.AUG.CONTRAST, 
                                                        saturation=cfg.AUG.SATURATION, 
                                                        hue=cfg.AUG.HUE),
                                        CDMToTensor(),
                                        CDMNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
                else:
                    trans = CDMCompose([CDMToTensor(),
                                        CDMNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
        elif cfg.FINETUNE.TYPE == 'det':
            if is_aug:
                trans = DETCompose([DETRandomHorizontalFlip(),
                                    DETRandomVerticalFlip(),
                                    DETResize(cfg.DATA.INPUT_SIZE),
                                    # DETRandomResizedCrop(size=cfg.DATA.INPUT_SIZE, 
                                    #                     scale=cfg.AUG.CROP_SCALE, 
                                    #                     ratio=cfg.AUG.CROP_RATIO),
                                    DETColorJitter(brightness=cfg.AUG.INTENSITY, 
                                                    contrast=cfg.AUG.CONTRAST, 
                                                    saturation=cfg.AUG.SATURATION, 
                                                    hue=cfg.AUG.HUE),
                                    # DETToTensor(),
                                    # DETNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
                                    DETToTensor()])

            else:
                # trans = DETCompose([DETToTensor(),
                #                     DETNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
                trans = DETCompose([DETResize(cfg.DATA.INPUT_SIZE),
                                    DETToTensor()])
                
        else:
            raise NotImplementedError('Not implemented finetune type: {}'.format(cfg.FINETUNE.TYPE))
        
    elif mode == 'Inference':
        if cfg.FINETUNE.TYPE == 'cd':
            trans = CDSCompose([CDSToTensor(),
                                CDSNormalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD)])
        else:
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])

    else:
        raise NotImplementedError('Not implemented mode: {}'.format(mode))
    return trans


def build_metrics(cfg, is_identity=False):
    if is_identity:
        metrics = IdentityMetrics()
    else:
        if cfg.FINETUNE.TYPE == 'ic':
            metrics = ImageClassificationMetrics(cfg.FINETUNE.IC.NUM_CLASSES, cfg.FINETUNE.IC.DEFAULT_METRIC, ignore_index=cfg.DATA.IGNORE_INDEX)
        elif cfg.FINETUNE.TYPE == 'cd':
            if cfg.FINETUNE.CD.NUM_CLASSES == 2:
                metrics = SemanticSegmentationMetrics(cfg.FINETUNE.CD.NUM_CLASSES, cfg.FINETUNE.CD.CLASS_LIST, cfg.FINETUNE.CD.DEFAULT_METRIC, ignore_index=cfg.DATA.IGNORE_INDEX)
            else:
                metrics = ChangeDetectionMetrics(cfg.FINETUNE.CD.NUM_CLASSES, cfg.FINETUNE.CD.CLASS_LIST, cfg.FINETUNE.CD.DEFAULT_METRIC, cfg.FINETUNE.CD.MASK_ID)
        elif cfg.FINETUNE.TYPE == 'seg':
            metrics = SemanticSegmentationMetrics(cfg.FINETUNE.SEG.NUM_CLASSES, cfg.FINETUNE.SEG.CLASS_LIST, cfg.FINETUNE.SEG.DEFAULT_METRIC, ignore_index=cfg.DATA.IGNORE_INDEX)
        elif cfg.FINETUNE.TYPE == 'det':
            metrics = ObjectDetectionMetrics(cfg.FINETUNE.DET.DEFAULT_METRIC)
        else:
            raise NotImplementedError(f"{cfg.FINETUNE.TYPE} is not implemented yet.")
    return metrics


def build_optimizer(cfg, params_groups):
    if cfg.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam([{'params': params_groups}], 
                                     lr=cfg.LR_SCHEDULER.WARMUP_VALUE, 
                                     betas=cfg.OPTIMIZER.BETAS, 
                                     weight_decay=cfg.WD_SCHEDULER.WARMUP_VALUE, 
                                     eps=cfg.OPTIMIZER.EPS,
                                     fused=True)
    elif cfg.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD([{'params': params_groups}], 
                                     lr=cfg.LR_SCHEDULER.WARMUP_VALUE, 
                                     weight_decay=cfg.WD_SCHEDULER.WARMUP_VALUE,
                                     momentum=cfg.OPTIMIZER.MOMENTUM)
    elif cfg.OPTIMIZER.NAME == 'adamw':
        optimizer = torch.optim.AdamW([{'params': params_groups}], 
                                     lr=cfg.LR_SCHEDULER.WARMUP_VALUE, 
                                     betas=cfg.OPTIMIZER.BETAS, 
                                     weight_decay=cfg.WD_SCHEDULER.WARMUP_VALUE, 
                                     eps=cfg.OPTIMIZER.EPS,
                                     fused=True)
    elif cfg.OPTIMIZER.NAME == 'rmsprop':
        optimizer = torch.optim.RMSprop([{'params': params_groups}], 
                                        lr=cfg.LR_SCHEDULER.WARMUP_VALUE, 
                                        momentum=cfg.OPTIMIZER.MOMENTUM, 
                                        weight_decay=cfg.WD_SCHEDULER.WARMUP_VALUE,
                                        eps=cfg.OPTIMIZER.EPS)
    elif cfg.OPTIMIZER.NAME == 'lars':
        optimizer = LARS([{'params': params_groups}], 
                            lr=cfg.LR_SCHEDULER.WARMUP_VALUE, 
                            momentum=cfg.OPTIMIZER.MOMENTUM, 
                            weight_decay=cfg.WD_SCHEDULER.WARMUP_VALUE)
    else:
        raise NotImplementedError(f"Unkown optimizer: {cfg.OPTIMIZER.NAME}")
    return optimizer


def build_schedulers(cfg, epoch_length, is_pretrain=True):
    lr = dict(
        start_value=cfg.LR_SCHEDULER.LEARNING_RATE,
        final_value=cfg.LR_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.NUM_EPOCHS * epoch_length,
        warmup_iters=cfg.LR_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.LR_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.LR_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.LR_SCHEDULER.IS_RESTART,
        T_0=cfg.LR_SCHEDULER.T_0,
        T_mult=cfg.LR_SCHEDULER.T_MULT,
    )
    wd = dict(
        start_value=cfg.WD_SCHEDULER.WEIGHT_DECAY,
        final_value=cfg.WD_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.NUM_EPOCHS * epoch_length,
        warmup_iters=cfg.WD_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.WD_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.WD_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.WD_SCHEDULER.IS_RESTART,
        T_0=cfg.WD_SCHEDULER.T_0,
        T_mult=cfg.WD_SCHEDULER.T_MULT,
    )
    momentum = dict(
        start_value=cfg.TM_SCHEDULER.TEACHER_MOMENTUM,
        final_value=cfg.TM_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.NUM_EPOCHS * epoch_length,
        warmup_iters=cfg.TM_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.TM_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.TM_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.TM_SCHEDULER.IS_RESTART,
        T_0=cfg.TM_SCHEDULER.T_0,
        T_mult=cfg.TM_SCHEDULER.T_MULT,
    )
    teacher_temp = dict(
        start_value=cfg.TT_SCHEDULER.TEACHER_TEMP,
        final_value=cfg.TT_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.TT_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        warmup_iters=cfg.TT_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.TT_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.TT_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.TT_SCHEDULER.IS_RESTART,
        T_0=cfg.TT_SCHEDULER.T_0,
        T_mult=cfg.TT_SCHEDULER.T_MULT,
    )

    ft_lr = dict(
        start_value=cfg.LR_SCHEDULER.LEARNING_RATE,
        final_value=cfg.LR_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.FINETUNE.NUM_EPOCHS * epoch_length,
        warmup_iters=cfg.LR_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.LR_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.LR_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.LR_SCHEDULER.IS_RESTART,
        T_0=cfg.LR_SCHEDULER.T_0,
        T_mult=cfg.LR_SCHEDULER.T_MULT,
    )
    ft_wd = dict(
        start_value=cfg.WD_SCHEDULER.WEIGHT_DECAY,
        final_value=cfg.WD_SCHEDULER.FINAL_VALUE,
        total_iters=cfg.FINETUNE.NUM_EPOCHS * epoch_length,
        warmup_iters=cfg.WD_SCHEDULER.WARMUP_EPOCHS * epoch_length,
        start_warmup_value=cfg.WD_SCHEDULER.WARMUP_VALUE,
        freeze_iters=cfg.WD_SCHEDULER.FREEZE_EPOCHS * epoch_length,
        is_restart=cfg.WD_SCHEDULER.IS_RESTART,
        T_0=cfg.WD_SCHEDULER.T_0,
        T_mult=cfg.WD_SCHEDULER.T_MULT,
    )

    if is_pretrain:
        lr_scheduler = CosineScheduler(**lr)
        wd_scheduler = CosineScheduler(**wd)
        momentum_scheduler = CosineScheduler(**momentum)
        teacher_temp_scheduler = CosineScheduler(**teacher_temp)
        last_layer_lr_scheduler = CosineScheduler(**lr)

        last_layer_lr_scheduler.schedule[:cfg.FREEZE_LAST_LAYER_EPOCHS * epoch_length] = 0  # mimicking the original schedules

        return (
            lr_scheduler,
            wd_scheduler,
            momentum_scheduler,
            teacher_temp_scheduler,
            last_layer_lr_scheduler,
        )
    else:
        lr_scheduler = CosineScheduler(**ft_lr)
        wd_scheduler = CosineScheduler(**ft_wd)
        return lr_scheduler, wd_scheduler


def test_CosineScheduler():
    # test CosineScheduler class
    import matplotlib.pyplot as plt
    import numpy as np

    start_value = 0
    final_value = 1
    total_iters = 100
    warmup_iters = 10
    start_warmup_value = 0
    freeze_iters = 0
    is_restart = False
    T_0 = 5
    T_mult = 2
    scheduler = CosineScheduler(start_value, final_value, total_iters, warmup_iters, start_warmup_value, freeze_iters, is_restart, T_0, T_mult)
    x = np.arange(total_iters)
    y = np.zeros(total_iters)
    for i in range(total_iters):
        y[i] = scheduler[i]
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    test_CosineScheduler()