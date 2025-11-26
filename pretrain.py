import argparse
import torch
from tqdm import tqdm
from pretrain_frame import PretrainFrame
from Utils import Timer, build_logger, build_dataset, set_seed
from Utils.config import get_config, save_config, get_output_path, check_config
import torch.distributed as dist
import os


def pretrain(cfg, frame, pretrain_dataset, logger):
    total_timer = Timer()
    epoch_timer = Timer()
    total_timer.start()

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=global_rank,
                                                                    shuffle=True)
    train_data_loader = frame.get_data_loader(pretrain_dataset, train_sampler)

    if global_rank == 0:
        command_line = ' '.join(sys.argv)
        logger.info(f'Command: {command_line}')
        logger.info(f'World_size: {world_size}')
        logger.info(f'{pretrain_dataset.__len__()} examples in training set')
        logger.info(f"Number of params: {frame.num_params/ 1e6:.2f}M in model {cfg.NET.NAME}")
    dist.barrier()  # to sync logging that may cause hang, cause one process exits before the other one

    for epoch in range(cfg.START_EPOCH, cfg.NUM_EPOCHS):
        epoch_timer.start()
        train_sampler.set_epoch(epoch)
        train_data_loader_iter = iter(train_data_loader)
        train_epoch_loss = 0
        for images in tqdm(train_data_loader_iter, ncols=80):
            frame.set_input(images)
            l = frame.optimize()
            train_epoch_loss += l
        train_epoch_loss /= len(train_data_loader_iter)
        if frame.loss_dict:
            for k, v in frame.loss_dict.items():
                frame.loss_dict[k] = v / len(train_data_loader_iter)
        dist.barrier()

        # gather all losses
        with torch.no_grad():
            def sync_loss(loss):
                all_losses_list = [torch.zeros(1).cuda() for _ in range(world_size)]
                dist.all_gather(all_losses_list, loss.float().cuda())
                mean_loss = (sum(all_losses_list) / len(all_losses_list)).item()
                return mean_loss
            epoch_mean_loss = sync_loss(train_epoch_loss)
            if cfg.NET.NAME == 'dinov2' or cfg.NET.NAME == 'pera':
                loss_dict = {}
                for k, v in frame.loss_dict.items():
                    mean_loss = sync_loss(v)
                    loss_dict[k] = round(mean_loss, 3)
            frame.loss_dict.clear()
        dist.barrier()

        
        # save model
        if epoch == 0:
            frame.best_rank_loss = train_epoch_loss
            frame.best_mean_loss = epoch_mean_loss
        else:
            if epoch_mean_loss < frame.best_mean_loss:
                frame.best_mean_loss = epoch_mean_loss
            if train_epoch_loss < frame.best_rank_loss:
                frame.best_rank_loss = train_epoch_loss
        if (epoch  + 1) % cfg.SAVE_FREQ == 0 or epoch == cfg.NUM_EPOCHS - 1:
            frame.save_weights(cfg.OUTPUT_PATH, cfg.NET.NAME, cfg.CFG_NOTE, epoch, epoch_mean_loss)
        dist.barrier()

        # print info
        epoch_timer.stop()
        if cfg.NET.NAME == 'mae':
            logger.debug(f'[Rank {global_rank}] epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                    f'train_loss: {train_epoch_loss:.3f}, best_loss: {frame.best_rank_loss:.3f}, '
                    f'lr: {frame.learning_rate:.2e}, wd:{frame.weight_decay:.2e}')
        else:
            logger.debug(f'[Rank {global_rank}] epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                        f'train_loss: {train_epoch_loss:.3f}, best_loss: {frame.best_rank_loss:.3f}, '
                        f'lr: {frame.learning_rate:.2e}, wd:{frame.weight_decay:.2e}, '
                        f'teacher_temp:{frame.teacher_temperature:.3f}, teacher_mom:{frame.teacher_momentum:.3f}')
        if global_rank == 0:
            if cfg.NET.NAME == 'mae':
                logger.info(f'epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                        f'mean_loss: {epoch_mean_loss:.3f}, best_mean_loss: {frame.best_mean_loss:.3f}, '
                        f'lr: {frame.learning_rate:.2e}, wd:{frame.weight_decay:.2e}')
            else:
                logger.info(f'epoch: {epoch}, epoch_time: {epoch_timer.get_epochtime()}, '
                        f'mean_loss: {epoch_mean_loss:.3f}, best_mean_loss: {frame.best_mean_loss:.3f}, '
                        f'lr: {frame.learning_rate:.2e}, wd:{frame.weight_decay:.2e}, '
                        f'teacher_temp:{frame.teacher_temperature:.3f}, teacher_mom:{frame.teacher_momentum:.3f}')
            if cfg.NET.NAME == 'dinov2' or cfg.NET.NAME == 'pera':
                logger.info(", ".join([f'{k}: {v}' for k, v in loss_dict.items()]) + '\n')
        dist.barrier()
    
    total_timer.stop()
    logger.info(f'[Rank {global_rank}] train_time: {epoch_timer.get_sumtime()}, '
                  f'{pretrain_dataset.__len__() * (cfg.NUM_EPOCHS - cfg.START_EPOCH) / epoch_timer.sum():.2f}examples/sec, '
                  f'total_time: {total_timer.get_sumtime()}, best_mean_loss: {frame.best_mean_loss:.3f}')
    logger.info('[Rank {global_rank}] Finish!')


def get_parserargs():
    parser = argparse.ArgumentParser(description='Train the network on images using Pytorch')

    # ==========training misc setting==========
    # config setting
    parser.add_argument('--cfg_path', '-cfg', type=str, default=None, metavar="CFG", help='path to load a local config file')
    parser.add_argument('--cfg_note', '-cn', metavar='CN', type=str, help='note which will be saved in config name')
    parser.add_argument('--is_resume', '-r', metavar='R', default=False, help="""whether to resume training, if false and pretrain_path is not None,
                        will train on pretrain model but from epoch 0""")
    parser.add_argument('--pretrain_path', '-pp', metavar='PP', type=str, default=None, help='pretrain model abspath')
    parser.add_argument('--output_path', '-op', type=str, help='output dir to save log, model, cfg...')
    parser.add_argument('--seed', '-s', type=int, help='random seed')

    # train setting
    parser.add_argument('--save_freq', '-sf', type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--num_epochs', '-e', metavar='E', type=int, help='Number of training epochs')
    parser.add_argument('--freeze_last_layer_epochs', '-flle', type=int, help="""Number of epochs during which we keep the output layer fixed. 
                        Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--is_benchmark', '-ib', help="""Whether to use cudnn benchmark, if true, cudnn will find the best algorithm to use for your hardware and 
                        the result may be irreducible.""")
    parser.add_argument('--dtype', '-t',type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'], help='Training data type')
    parser.add_argument('--gradient_clipping', '-gc', type=float, help="""Maximal parameter gradient norm if using gradient clipping. 
                        Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--gradient_accumulation_steps', '-gas', type=int, help="""Number of steps before performing a backward/update pass.""")
    
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument("--local_rank", type=int, help="Please ignore and do not set this argument.")
    # ==============data setting=============
    parser.add_argument('--input_size', '-is', metavar='IS', type=int, help='Input size of image')
    parser.add_argument('--pretrain_data_path', '-pdp', type=str, help='Pretrain dataset abspath for pretraining')
    parser.add_argument('--num_workers', '-w', metavar='NW', type=int, help='number of workers in dataloader')
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, help='Batch size per gpu')
    parser.add_argument('--is_pin_memory', '-pm', metavar='PM', default=True, help='is pin memory in dataloader')

    # ==========augmentation setting=========
    parser.add_argument('--is_aug', '-a', metavar='AUG', help='Use augmentations or not')
    
    parser.add_argument('--aug_normalize_mean', '-anm', type=float, nargs='+', help="""Mean of the normalization.""")
    parser.add_argument('--aug_normalize_std', '-ans', type=float, nargs='+', help="""Std of the normalization.""")

    # Color jitter factor setting
    parser.add_argument('--aug_intensity', '-ai', metavar='AI', type=float, help='intensity of color jitter')
    parser.add_argument('--aug_hue', '-ah', metavar='AH', type=float, help='hue of color jitter')
    parser.add_argument('--aug_saturation', '-asa', metavar='ASA', type=float, help='saturation of color jitter')
    parser.add_argument('--aug_contrast', '-ac', metavar='AC', type=float, help='contrast of color jitter')
    
    # Multi-crop setting
    parser.add_argument('--aug_num_global', '-ang', type=int, help="""Number of large global views to generate""")
    parser.add_argument('--aug_global_crop_size', '-agc', type=int, help="""Size of the large global views. Used for multi-crop training.""")
    parser.add_argument('--aug_global_scale', '-ags', type=float, nargs='+', help="""Scale range of the cropped image before resizing, 
                        relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we 
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--aug_global_ratio', '-agr', type=float, nargs='+', help="""Ratio range of the cropped image before resizing""")
    parser.add_argument('--aug_num_local', '-anl', type=int, help="""Number of small local views to generate. Set this parameter to 0 to disable
                         multi-crop training. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--aug_local_crop_size', '-alc', type=int, help="""Size of the small local views. Used for multi-crop training.""")
    parser.add_argument('--aug_local_scale', '-als', type=float, nargs='+', help="""Scale range of the cropped image before resizing, 
                        relatively to the origin image. Used for small local view cropping of multi-crop.""")
    parser.add_argument('--aug_local_ratio', '-alr', type=float, nargs='+', help="""Ratio range of the cropped image before resizing""")


    # ==========net setting==========
    parser.add_argument('--net_name', '-n', metavar='N', type=str, help='Network name')
    parser.add_argument('--net_dropout_rate', '-ndr', metavar='NDR', type=float, help='dropout rate')
    parser.add_argument('--net_dropout_path_rate', '-ndpr', metavar='NDPR', type=float, help='stochastic depth dropout path rate')
    parser.add_argument('--net_patch_size', '-nps', metavar='NPS', type=int, help="""Using smaller values leads to better performance but requires more memory. 
                        If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--net_embed_dim', '-ned', type=int, help='Embed dim')
    parser.add_argument('--net_depth', '-nd', type=int, help='Depth')
    parser.add_argument('--net_num_heads', '-nnh', type=int, help='Number of attention heads')
    parser.add_argument('--net_mlp_ratio', '-nmr', type=float, help='mlp ratio')
    parser.add_argument('--net_in_chans', '-nic', type=int, help='Input channels for the network, 3 for RGB, 1 for grayscale')

    
    # ==========optimization setting==========
    # optimizer setting
    parser.add_argument('--optim_name', '-o', metavar='O', type=str, help='Optimizer name, sgd, adam, rmsprop, adamw...')
    parser.add_argument('--optim_momentum', '- omm', metavar='OMM', type=float, help='beta of optimize momentum vt, some optimizer do not need')
    parser.add_argument('--optim_eps', '-oe', metavar='OE', type=float, help='eps of adam, rmsprop, adamw...')
    parser.add_argument('--optim_betas', '-obt', metavar='OBT', type=float, nargs=2, help='betas of adam, adamw...')

    # lr scheduler setting
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, help='Learning rate (start with max learning rate)')
    parser.add_argument('--lrs_final_value', '-lrsfv', metavar='LRSMV', type=float, help='final value of cosinewarm scheduler(min in cycle)')
    parser.add_argument('--lrs_warmup_epochs', '-lrswe', metavar='LRSWI', type=int, help='warmup epochs of cosinewarm scheduler')
    parser.add_argument('--lrs_warmup_value', '-lrswv', metavar='LRSSWV', type=float, help='start warmup value of cosinewarm scheduler')
    parser.add_argument('--lrs_freeze_epochs', '-lrsfe', metavar='LRSFI', type=int, help='freeze epochs of cosinewarm scheduler')
    parser.add_argument('--lrs_is_restart', '-lrsir', metavar='LRSIR', default=False, help='is restart of cosinewarm scheduler')
    parser.add_argument('--lrs_T_0', '-lrst0', metavar='LRSTMA', type=int, help='T_0 of cosinewarm scheduler if restart')
    parser.add_argument('--lrs_T_mult', '-lrstmu', metavar='LRSTMU', type=int, help='T_mult of cosinewarm scheduler if restart')

    # weight decay scheduler setting
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a 
                        smaller value at the beginning of training works well.""")
    parser.add_argument('--wds_final_value', '-wdsfv', type=float, help='final value of cosinewarm scheduler(max in cycle)')
    parser.add_argument('--wds_warmup_epochs', '-wdswe', type=int, help='warmup epochs of cosinewarm scheduler')
    parser.add_argument('--wds_warmup_value', '-wdswv', type=float, help='start warmup value of cosinewarm scheduler')
    parser.add_argument('--wds_freeze_epochs', '-wdsfe', default=0, type=int, help='freeze epochs of cosinewarm scheduler')
    parser.add_argument('--wds_is_restart', '-wdsir', default=False, help='is restart of cosinewarm scheduler')
    parser.add_argument('--wds_T_0', '-wdst0', type=int, help='T_0 of cosinewarm scheduler if restart')
    parser.add_argument('--wds_T_mult', '-wdstmu', type=int, help='T_mult of cosinewarm scheduler if restart')

    # teacher momentum scheduler setting
    parser.add_argument('--teacher_momentum', '-tm', metavar='TM', type=float, help="""Base EMA parameter for teacher update. 
                        The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: 
                        for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--tms_final_value', '-tmsfv', metavar='TMSFV', type=float, help='Final value for the teacher momentum.')
    
    # teacher temperature scheduler setting
    parser.add_argument('--student_temp', '-st', type=float, help="Initial value for the student temperature: 0.1 works well in most cases.")
    parser.add_argument('--teacher_temp', '-tt', type=float, help="""Final value (after linear warmup) of the teacher temperature. For most experiments, 
                        anything above 0.07 is unstable. We recommendstarting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--tts_final_value', "-ttsfv", metavar='TTSFV', type=float, help="""Final value (after linear warmup) of the teacher temperature. 
                        For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and 
                        increase this slightly if needed.""")
    parser.add_argument('--tts_warmup_value', '-ttswv', type=float, help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--tts_warmup_epochs', '-ttswe', type=int, help='warmup epochs of cosinewarm scheduler')

    args, unknown = parser.parse_known_args()

    config = get_config(args)
    return args, config



if __name__ == '__main__':
    args, cfg = get_parserargs()
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"RANK and WORLD_SIZE in environ: {global_rank}/{world_size}")
    else:
        global_rank = -1
        world_size = -1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=global_rank)
    dist.barrier()

    # output path
    cfg.defrost()
    length = torch.tensor(0).cuda()
    if dist.get_rank() == 0:
        cfg.OUTPUT_PATH = get_output_path(cfg)
        save_config(cfg)
        path_tensor = torch.ByteTensor(list(cfg.OUTPUT_PATH.encode('utf-8'))).cuda()
        length = torch.tensor(len(path_tensor)).cuda()
    dist.barrier()
    dist.broadcast(length, src=0)
    dist.barrier()
    if dist.get_rank() != 0:
        path_tensor = torch.ByteTensor([0] * length.item()).cuda()
    dist.broadcast(path_tensor, src=0)
    cfg.OUTPUT_PATH = bytes(path_tensor.tolist()).decode('utf-8')
    cfg.freeze()
    dist.barrier()

    logger = build_logger(cfg.OUTPUT_PATH, cfg.NET.NAME, cfg.CFG_NOTE, dist.get_rank())
    if dist.get_rank() == 0:
        check_config(cfg, logger)
    dist.barrier()

    set_seed(cfg.SEED + dist.get_rank(), cfg.IS_BENCHMARK)

    pretrain_dataset = build_dataset(cfg, data_path=cfg.DATA.PRETRAIN_DATA_PATH, mode='Pretrain', is_aug=True)
    frame = PretrainFrame(local_rank, cfg, pretrain_dataset.__len__(), logger)
    dist.barrier()
    pretrain(cfg, frame, pretrain_dataset, logger)
