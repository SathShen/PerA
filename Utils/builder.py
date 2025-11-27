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
        raise NotImplementedError(f"Not implemented")


def build_loss(cfg, is_pretrain=True):
    if is_pretrain == True:
        if cfg.LOSS.NAME == 'dinov1' or cfg.LOSS.NAME == 'dino':
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
        raise NotImplementedError(f"Not implemented")
    return loss_func


def build_transform(cfg, mode='Pretrain', is_aug=True):
    if mode == 'Pretrain':
        if is_aug == False:
            trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.AUG.NORMALIZE.MEAN, std=cfg.AUG.NORMALIZE.STD, inplace=True)])
        else:
            if cfg.NET.NAME == 'dinov1' or cfg.NET.NAME == 'dino':
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
    else:
        raise NotImplementedError('Not implemented mode: {}'.format(mode))
    return trans


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