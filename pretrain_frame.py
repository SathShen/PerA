import os
import sys
import math
import torch
import torch.nn as nn
from Utils import build_net, build_loss, build_optimizer, build_schedulers
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from Utils.collate import *
from functools import partial

class PretrainFrame():
    def __init__(self, local_rank, cfg, num_dataset, logger):
        self.cfg = cfg
        self.num_iter_per_epoch = num_dataset // (cfg.DATA.BATCH_SIZE_PER_GPU * dist.get_world_size())# 训练一个batch即一个iter
        self.epoch = cfg.START_EPOCH
        self.start_iter = cfg.START_EPOCH * self.num_iter_per_epoch
        self.it = self.start_iter
        self.logger = logger
        
        if cfg.DTYPE == 'fp16':
            self.dtype = torch.half
        elif cfg.DTYPE == 'bf16':
            self.dtype = torch.bfloat16
        else:
            raise NotImplementedError(f"Unrecognized dtype '{cfg.DTYPE}'")
        self.clip_grad = cfg.GRADIENT_CLIPPING
        self.grad_accum_steps = cfg.GRADIENT_ACCUMULATION_STEPS
        self.best_rank_loss = 6657
        self.best_mean_loss = 6657
        self.best_knn_top1acc = -1
        self.best_knn_top5acc = -1

        self.net = build_net(cfg, is_pretrain=True)
        self.num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        if dist.get_world_size() > 1:
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        self.net = self.net.cuda()
        self.net_without_ddp = self.net
        self.scaler = GradScaler()
        self.optimizer = build_optimizer(cfg, self.net.parameters())
        self.loss_fuc = build_loss(cfg, is_pretrain=True)
        (self.lr_scheduler,
        self.wd_scheduler,
        self.momentum_scheduler,
        self.teacher_temp_scheduler,
        self.last_layer_lr_scheduler) = build_schedulers(cfg, self.num_iter_per_epoch)
        self.net = DistributedDataParallel(self.net, device_ids=[local_rank])

        if cfg.PRETRAIN_PATH:
            self.load_weights(cfg.PRETRAIN_PATH)

        self.loss_dict = {}


    def get_data_loader(self, train_dataset, train_sampler):
        if self.cfg.NET.NAME == 'dinov2':
            patch_resolution = self.cfg.AUG.GLOBAL_CROP_SIZE // self.cfg.NET.PATCH_SIZE

            mask_generator = MaskingGenerator(input_size=(patch_resolution, patch_resolution), 
                                            max_num_patches=0.5 * patch_resolution * patch_resolution)
            collate_fn = partial(
                dinov2_collate,
                mask_ratio_tuple=self.cfg.NET.DINO.MASK_RATIO_TUPLE,
                mask_probability=self.cfg.NET.DINO.MASK_PROBABILITY,
                dtype=self.dtype,
                n_tokens=patch_resolution * patch_resolution,
                mask_generator=mask_generator)
            train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.cfg.DATA.BATCH_SIZE_PER_GPU,
                                                            sampler=train_sampler,
                                                            num_workers=self.cfg.DATA.NUM_WORKERS,
                                                            pin_memory=self.cfg.DATA.IS_PIN_MEMORY,
                                                            drop_last=True,
                                                            collate_fn=collate_fn)
        elif self.cfg.NET.NAME == 'pera':
            train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.cfg.DATA.BATCH_SIZE_PER_GPU,
                                                            sampler=train_sampler,
                                                            num_workers=self.cfg.DATA.NUM_WORKERS,
                                                            pin_memory=self.cfg.DATA.IS_PIN_MEMORY,
                                                            drop_last=True,
                                                            collate_fn=pera_collate)
        else:
            train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=self.cfg.DATA.BATCH_SIZE_PER_GPU,
                                                            sampler=train_sampler,
                                                            num_workers=self.cfg.DATA.NUM_WORKERS,
                                                            pin_memory=self.cfg.DATA.IS_PIN_MEMORY,
                                                            drop_last=True)
        return train_data_loader


    def apply_optim_scheduler(self, optimizer, lr, wd):
        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = wd
            param_group["lr"] = lr


    def set_input(self, images):
        if self.cfg.NET.NAME == 'dinov2':
            images_dict = {}
            images_dict["global_crops"] = images["collated_global_crops"].cuda(non_blocking=True).to(self.dtype)
            images_dict["local_crops"] = images["collated_local_crops"].cuda(non_blocking=True).to(self.dtype)
            images_dict["masks"] = images["collated_masks"].cuda(non_blocking=True)
            images_dict["mask_indices_list"] = images["mask_indices_list"].cuda(non_blocking=True)
            images_dict["num_masked"] = images["n_masked_patches"].cuda(non_blocking=True)
            images_dict["upperbound"] = images["upperbound"]
            images_dict["masks_weight"] = images["masks_weight"].cuda(non_blocking=True).to(self.dtype)
            self.images = images_dict
        elif self.cfg.NET.NAME == 'dinov1' or self.cfg.NET.NAME == 'dino':
            self.images = [image.cuda(non_blocking=True).to(self.dtype) for image in images]
        elif self.cfg.NET.NAME == 'pera':
            images_dict = {}
            images_dict["S_global_crops"] = images["S_collated_global_crops"].cuda(non_blocking=True).to(self.dtype)
            images_dict["T_global_crops"] = images["T_collated_global_crops"].cuda(non_blocking=True).to(self.dtype)
            images_dict["S_local_crops"] = images["S_collated_local_crops"].cuda(non_blocking=True).to(self.dtype)
            images_dict["T_local_crops"] = images["T_collated_local_crops"].cuda(non_blocking=True).to(self.dtype)
            self.images = images_dict
        elif self.cfg.NET.NAME == 'mocov3':
            imagesQ = images[0].cuda(non_blocking=True).to(self.dtype)
            imagesK = images[1].cuda(non_blocking=True).to(self.dtype)
            self.images = (imagesQ, imagesK)
        elif self.cfg.NET.NAME == 'mae':
            self.images = images.cuda(non_blocking=True).to(self.dtype)
        else:
            raise NotImplementedError(f"Input setting for {self.cfg.NET.NAME} is not implemented")

    def optimize(self):
        self.epoch = self.it // self.num_iter_per_epoch

        # apply schedules
        self.learning_rate = self.lr_scheduler[self.it]
        self.weight_decay = self.wd_scheduler[self.it]
        self.teacher_temperature = self.teacher_temp_scheduler[self.it]
        self.teacher_momentum = self.momentum_scheduler[self.it]
        self.apply_optim_scheduler(self.optimizer, self.learning_rate, self.weight_decay)


        # forward and backward
        with autocast(dtype=self.dtype):
            loss_dict = None
            student_output, teacher_output = self.net(self.images)
            if self.cfg.NET.NAME == 'dinov2':
                loss, loss_dict = self.loss_fuc(student_output, teacher_output, self.teacher_temperature, self.images)
            elif self.cfg.NET.NAME == 'pera':
                loss, loss_dict = self.loss_fuc(student_output, teacher_output, self.teacher_temperature)
            elif self.cfg.NET.NAME == 'mae':
                loss = self.loss_fuc(student_output)
            else:
                loss = self.loss_fuc(student_output, teacher_output, self.teacher_temperature)
            if loss_dict is not None:
                for k, v in loss_dict.items():
                    self.loss_dict[k] = self.loss_dict[k] + v.detach() if k in self.loss_dict else v
            if self.grad_accum_steps is not None and self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps
        self.scaler.scale(loss).backward()

        if (self.it + 1) % self.grad_accum_steps == 0 or (self.it + 1) == self.num_iter_per_epoch:
            self.scaler.unscale_(self.optimizer)
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.net.module.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # check if loss is valid
        if not math.isfinite(loss.item()):
            self.logger.warning(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        # EMA update for the teacher
        if self.cfg.NET.NAME != 'mae':
            with torch.no_grad():
                for param_q, param_k in zip(self.net.module.student.parameters(), self.net.module.teacher.parameters()):
                    param_k.data.mul_(self.teacher_momentum).add_((1 - self.teacher_momentum) * param_q.detach().data)
        self.it += 1
        return loss.detach()
            

    def save_weights(self, output_path, net_name, cfg_note, epoch, loss):
        if dist.get_rank() == 0:
            output_path = output_path + '/autosave'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            path = f"{output_path}/{net_name}_{cfg_note}_ep{epoch}_auto.params"
            save_state = {'model': self.net_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_loss': loss,
            'epoch': epoch}
            torch.save(save_state, path)
            self.logger.info(f"Model saved to {path}")

            
    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.net_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.best_mean_loss = checkpoint['best_loss']
        del checkpoint
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

