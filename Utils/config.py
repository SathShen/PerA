import os
import yaml
import time
from yacs.config import CfgNode as CN
import argparse
import sys

# <editor-fold desc="base config setting">
base_cfg = CN(new_allowed=True)
# Base config files
base_cfg.BASE = ['']

# -----------------------------------------------------------------------------
# training misc
# -----------------------------------------------------------------------------
base_cfg.CFG_PATH = None
base_cfg.CFG_NOTE = 'default'
base_cfg.IS_RESUME = False
base_cfg.PRETRAIN_PATH = None
base_cfg.OUTPUT_PATH = None

base_cfg.NUM_NODES = 1
base_cfg.NUM_GPUS_PER_NODE = 1
base_cfg.WORLD_SIZE = 1
base_cfg.SEED = 10
base_cfg.IS_BENCHMARK = True

base_cfg.SAVE_FREQ = 10
base_cfg.NUM_EPOCHS = 200
base_cfg.FREEZE_LAST_LAYER_EPOCHS = 0  # dinov1
base_cfg.DTYPE = 'fp16'
base_cfg.GRADIENT_CLIPPING = 3.0
base_cfg.GRADIENT_ACCUMULATION_STEPS = 1

# no need to set
base_cfg.START_EPOCH = 0
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
base_cfg.DATA = CN()
base_cfg.DATA.INPUT_SIZE = 512
base_cfg.DATA.PRETRAIN_DATA_PATH = ''
base_cfg.DATA.TRAIN_DATA_PATH = ''
base_cfg.DATA.VALID_DATA_PATH = ''
base_cfg.DATA.TEST_DATA_PATH = ''
base_cfg.DATA.NUM_WORKERS = 16
base_cfg.DATA.BATCH_SIZE_PER_GPU = 16
base_cfg.DATA.IS_PIN_MEMORY = True
base_cfg.DATA.IGNORE_INDEX = None

# Interpolation to resize image (random, bilinear, bicubic)
base_cfg.DATA.INTERPOLATION = 'bilinear'

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
base_cfg.AUG = CN()

base_cfg.AUG.IS_AUG = True
base_cfg.AUG.NORMALIZE = CN()
base_cfg.AUG.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
base_cfg.AUG.NORMALIZE.STD = [0.229, 0.224, 0.225]

# Random crop resize
base_cfg.AUG.CROP_SCALE = (0.4, 1.)
base_cfg.AUG.CROP_RATIO = (0.75, 1.33)


# Color jitter factor
base_cfg.AUG.INTENSITY = 0.4
base_cfg.AUG.HUE = 0.1
base_cfg.AUG.SATURATION = 0.2
base_cfg.AUG.CONTRAST = 0.4

# Multi-crop for unsupervised learning
base_cfg.AUG.NUM_GLOBAL = 2
base_cfg.AUG.GLOBAL_CROP_SIZE = 448  # for the input size = 512
base_cfg.AUG.GLOBAL_SCALE = (0.4, 1.)
base_cfg.AUG.GLOBAL_RATIO = (0.75, 1.33)
base_cfg.AUG.NUM_LOCAL = 8
base_cfg.AUG.LOCAL_CROP_SIZE = 96
base_cfg.AUG.LOCAL_SCALE = (0.05, 0.4)
base_cfg.AUG.LOCAL_RATIO = (0.75, 1.33)

# -----------------------------------------------------------------------------
# Net settings
# -----------------------------------------------------------------------------
base_cfg.NET = CN()
base_cfg.NET.NAME = 'pera'
base_cfg.NET.DROP_RATE = 0.0
base_cfg.NET.DROP_PATH_RATE = 0.3
base_cfg.NET.DEPTH = 40
base_cfg.NET.PATCH_SIZE = 16
base_cfg.NET.EMBED_DIM = 1024

base_cfg.NET.NUM_HEADS = 16
base_cfg.NET.MLP_RATIO = 4.                # mlp hidden layers ratio
base_cfg.NET.IN_CHANS = 3


# Dino parameters
base_cfg.NET.DINO = CN()
base_cfg.NET.DINO.IS_NORM_LAST_LAYER = True
base_cfg.NET.DINO.IS_BN_IN_HEAD = False

base_cfg.NET.DINO.MASK_PROBABILITY = 0.5
base_cfg.NET.DINO.MASK_RATIO_TUPLE = (0.1, 0.5)
base_cfg.NET.DINO.NUM_HEAD_LAYERS = 3
base_cfg.NET.DINO.HEAD_OUT_DIM = 131072
base_cfg.NET.DINO.HEAD_HIDDEN_DIM = 2048
base_cfg.NET.DINO.HEAD_BOTTLENECK_DIM = 384
base_cfg.NET.DINO.CENTERING = 'centering'

# pera parameters
base_cfg.NET.PERA = CN()
base_cfg.NET.PERA.S_RATIO = 0.2
base_cfg.NET.PERA.T_RATIO = 0.6
base_cfg.NET.PERA.DINO_LOSS_WEIGHT = 1.0
# base_cfg.NET.PERA.IBOT_LOSS_WEIGHT = 1.0
base_cfg.NET.PERA.KOLEO_LOSS_WEIGHT = 0.1
base_cfg.NET.PERA.MAE_LOSS_WEIGHT = 0.01

# moco v3 parameters
base_cfg.NET.MOCO = CN()
base_cfg.NET.MOCO.HEAD_HIDDEN_DIM = 4096
base_cfg.NET.MOCO.HEAD_OUT_DIM = 256


# -----------------------------------------------------------------------------
# Optimization settings
# -----------------------------------------------------------------------------
base_cfg.LOSS = CN()
base_cfg.LOSS.NAME = 'focal'
base_cfg.LOSS.IS_AVERAGE = True


# Optimizer
base_cfg.OPTIMIZER = CN()
base_cfg.OPTIMIZER.NAME = 'adamw'
# SGD momentum
base_cfg.OPTIMIZER.MOMENTUM = 0.9
# Optimizer Epsilon
base_cfg.OPTIMIZER.EPS = 2e-5
# Adam Optimizer Betas
base_cfg.OPTIMIZER.BETAS = (0.9, 0.999)



# lr scheduler setting
base_cfg.LR_SCHEDULER = CN()
base_cfg.LR_SCHEDULER.LEARNING_RATE = 1e-3
base_cfg.LR_SCHEDULER.FINAL_VALUE = 1e-6
base_cfg.LR_SCHEDULER.WARMUP_EPOCHS = 20
base_cfg.LR_SCHEDULER.WARMUP_VALUE = 0
base_cfg.LR_SCHEDULER.FREEZE_EPOCHS = 0
base_cfg.LR_SCHEDULER.IS_RESTART = False
base_cfg.LR_SCHEDULER.T_0 = 10
base_cfg.LR_SCHEDULER.T_MULT = 2

# weight decay scheduler setting
base_cfg.WD_SCHEDULER = CN()
base_cfg.WD_SCHEDULER.WEIGHT_DECAY = 0.04
base_cfg.WD_SCHEDULER.FINAL_VALUE = 0.2
base_cfg.WD_SCHEDULER.WARMUP_EPOCHS = 0
base_cfg.WD_SCHEDULER.WARMUP_VALUE = 0.
base_cfg.WD_SCHEDULER.FREEZE_EPOCHS = 0
base_cfg.WD_SCHEDULER.IS_RESTART = False
base_cfg.WD_SCHEDULER.T_0 = 10
base_cfg.WD_SCHEDULER.T_MULT = 2

# teacher momentum scheduler setting
base_cfg.TM_SCHEDULER = CN()
base_cfg.TM_SCHEDULER.TEACHER_MOMENTUM = 0.998
base_cfg.TM_SCHEDULER.FINAL_VALUE = 1.
base_cfg.TM_SCHEDULER.WARMUP_EPOCHS = 0
base_cfg.TM_SCHEDULER.WARMUP_VALUE = 0.
base_cfg.TM_SCHEDULER.FREEZE_EPOCHS = 0
base_cfg.TM_SCHEDULER.IS_RESTART = False
base_cfg.TM_SCHEDULER.T_0 = 10
base_cfg.TM_SCHEDULER.T_MULT = 2

base_cfg.STUDENT_TEMP = 0.1
# teacher temperature scheduler setting\
base_cfg.TT_SCHEDULER = CN()
base_cfg.TT_SCHEDULER.TEACHER_TEMP = 0.03
base_cfg.TT_SCHEDULER.FINAL_VALUE = 0.03
base_cfg.TT_SCHEDULER.WARMUP_EPOCHS = 30
base_cfg.TT_SCHEDULER.WARMUP_VALUE = 0.02
base_cfg.TT_SCHEDULER.FREEZE_EPOCHS = 0
base_cfg.TT_SCHEDULER.IS_RESTART = False
base_cfg.TT_SCHEDULER.T_0 = 10
base_cfg.TT_SCHEDULER.T_MULT = 2



def bool_flag(str):
    """
    Parse boolean arguments from the command line.
    """
    if isinstance(str, bool):
        return str
    else:
        FALSY_STRINGS = {"off", "false", "0"}
        TRUTHY_STRINGS = {"on", "true", "1"}
        if str.lower() in FALSY_STRINGS or str is False:
            return False
        elif str.lower() in TRUTHY_STRINGS or str is True:
            return True
        else:
            raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def none_flag(str):
    """
    Parse None arguments from the command line.
    """
    # 如果是数字
    if isinstance(str, int) or isinstance(str, float) or isinstance(str, bool):
        return str
    elif str.lower() == 'none' or str == '' or str is None or str.lower() == 'null':
        return None
    else:
        return str


def norm_path(path):
    path = path.replace('\\', '/')
    path = path.replace('\\\\', '/')
    path = path.replace('//', '/')
    if path[-1] == '/':
        path = path[:-1]
    return path


def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    config.merge_from_file(cfg_file)


def update_config(config, args):
    def _check_args(name):
        if hasattr(args, name) and (eval(f'args.{name}') is not None):
            return True
        return False
    
    config.defrost()
    
    if _check_args('cfg_path'):
        cfg_path = none_flag(args.cfg_path)
        if cfg_path != None:
            assert os.path.exists(cfg_path), "The config file does not exist. Program exiting!"
            _update_config_from_file(config, cfg_path)
            config.CFG_PATH = norm_path(cfg_path)

    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    # config setting
    if _check_args('cfg_note'):
        config.CFG_NOTE = none_flag(args.cfg_note)
    if _check_args('is_resume'):
        config.IS_RESUME = bool_flag(args.is_resume)
    if _check_args('pretrain_path'):
        pretrain_path = none_flag(args.pretrain_path)
        if pretrain_path == None:
            config.PRETRAIN_PATH = None
            config.START_EPOCH = 0
        else:
            config.PRETRAIN_PATH = norm_path(args.pretrain_path)
            if len(os.path.split(config.PRETRAIN_PATH)[-1].split('_')) == 4:
                if config.IS_RESUME:
                    config.START_EPOCH = int(os.path.split(config.PRETRAIN_PATH)[1].split('_')[2][2:]) + 1
                else:
                    config.START_EPOCH = 0
                config.NET.NAME = os.path.split(config.PRETRAIN_PATH)[1].split('_')[0].lower()
            else:
                config.START_EPOCH = 0
                
    if _check_args('output_path'):
        if _check_args('is_resume') and bool_flag(args.is_resume):
            pass
        else:
            output_path = none_flag(args.output_path)
            if output_path != None:
                config.OUTPUT_PATH = norm_path(output_path)
            else:
                config.OUTPUT_PATH = None

    if _check_args('seed'):
        config.SEED = args.seed
    if _check_args('is_benchmark'):
        config.IS_BENCHMARK = bool_flag(args.is_benchmark)

    if _check_args('save_freq'):
        config.SAVE_FREQ = args.save_freq
    if _check_args('num_epochs'):
        config.NUM_EPOCHS = args.num_epochs
    if _check_args('freeze_last_layer_epochs'):
        config.FREEZE_LAST_LAYER_EPOCHS = args.freeze_last_layer_epochs


    # evaluate setting
    if _check_args('is_eval'):
        config.IS_EVAL = bool_flag(args.is_eval)
    if _check_args('is_save_pred'):
        config.IS_SAVE_PRED = bool_flag(args.is_save_pred)
    if _check_args('is_detailed'):
        config.IS_DETAILED = bool_flag(args.is_detailed)
    if _check_args('is_visualize_only'):
        config.IS_VISUALIZE_ONLY = bool_flag(args.is_visualize_only)
    if _check_args('is_inference'):
        config.IS_INFERENCE = bool_flag(args.is_inference)

    # data setting
    if _check_args('input_size'):
        config.DATA.INPUT_SIZE = args.input_size
    if _check_args('pretrain_data_path'):
        pretrain_data_path = none_flag(args.pretrain_data_path)
        if pretrain_data_path is None:
            config.DATA.PRETRAIN_DATA_PATH = None
        else:
            config.DATA.PRETRAIN_DATA_PATH = norm_path(pretrain_data_path)
    if _check_args('train_data_path'):
        train_data_path = none_flag(args.train_data_path)
        if train_data_path is None:
            config.DATA.TRAIN_DATA_PATH = None
        else:
            config.DATA.TRAIN_DATA_PATH = norm_path(train_data_path)
    if _check_args('valid_data_path'):
        valid_data_path = none_flag(args.valid_data_path)
        if valid_data_path is None:
            config.DATA.VALID_DATA_PATH = None
        else:
            config.DATA.VALID_DATA_PATH = norm_path(valid_data_path)
    if _check_args('test_data_path'):
        test_data_path = none_flag(args.test_data_path)
        if test_data_path is None:
            config.DATA.TEST_DATA_PATH = None
        else:
            config.DATA.TEST_DATA_PATH = norm_path(test_data_path)
    if _check_args('num_workers'):
        config.DATA.NUM_WORKERS = args.num_workers
    if _check_args('batch_size_per_gpu'):
        config.DATA.BATCH_SIZE_PER_GPU = args.batch_size_per_gpu
    if _check_args('is_pin_memory'):
        config.DATA.IS_PIN_MEMORY = bool_flag(args.is_pin_memory)
    if _check_args('ignore_index'):
        config.DATA.IGNORE_INDEX = args.ignore_index

    if _check_args('is_aug'):
        config.AUG.IS_AUG = bool_flag(args.is_aug)
    if _check_args('aug_normalize_mean'):
        config.AUG.NORMALIZE.MEAN = args.aug_normalize_mean
    if _check_args('aug_normalize_std'):
        config.AUG.NORMALIZE.STD = args.aug_normalize_std

    if _check_args('aug_crop_scale'):
        config.AUG.CROP_SCALE = args.aug_crop_scale
    if _check_args('aug_crop_ratio'):
        config.AUG.CROP_RATIO = args.aug_crop_ratio



    if _check_args('aug_intensity'):
        config.AUG.INTENSITY = args.aug_intensity
    if _check_args('aug_hue'):
        config.AUG.HUE = args.aug_hue
    if _check_args('aug_saturation'):
        config.AUG.SATURATION = args.aug_saturation
    if _check_args('aug_contrast'):
        config.AUG.CONTRAST = args.aug_contrast

    # Multi-crop setting
    if _check_args('aug_num_global'):
        config.AUG.NUM_GLOBAL = args.aug_num_global
    if _check_args('aug_global_crop_size'):
        config.AUG.GLOBAL_CROP_SIZE= args.aug_global_crop_size
    if _check_args('aug_global_scale'):
        config.AUG.GLOBAL_SCALE = args.aug_global_scale
    if _check_args('aug_global_ratio'):
        config.AUG.GLOBAL_RATIO = args.aug_global_ratio
    if _check_args('aug_num_local'):
        config.AUG.NUM_LOCAL = args.aug_num_local
    if _check_args('aug_local_crop_size'):
        config.AUG.LOCAL_CROP_SIZE = args.aug_local_crop_size
    if _check_args('aug_local_scale'):
        config.AUG.LOCAL_SCALE = args.aug_local_scale
    if _check_args('aug_local_ratio'):
        config.AUG.LOCAL_RATIO = args.aug_local_ratio

    
    # net setting
    if _check_args('net_name'):
        config.NET.NAME = none_flag(args.net_name.lower())
    if _check_args('net_dropout_rate'):
        config.NET.DROP_RATE = args.net_dropout_rate
    if _check_args('net_dropout_path_rate'):
        config.NET.DROP_PATH_RATE = args.net_dropout_path_rate
    if _check_args('net_patch_size'):
        config.NET.PATCH_SIZE = args.net_patch_size
    if _check_args('net_embed_dim'):
        config.NET.EMBED_DIM = args.net_embed_dim
    if _check_args('net_depth'):
        config.NET.DEPTH = args.net_depth
    if _check_args('net_num_heads'):
        config.NET.NUM_HEADS = args.net_num_heads
    if _check_args('net_mlp_ratio'):
        config.NET.MLP_RATIO = args.net_mlp_ratio
    if _check_args('net_in_chans'):
        config.NET.IN_CHANS = args.net_in_chans

    
    # dino
    if _check_args('net_dino_is_norm_last_layer'):      # only dinov1
        config.NET.DINO.IS_NORM_LAST_LAYER = bool_flag(args.net_dino_is_norm_last_layer)
    if _check_args('net_dino_is_bn_in_head'):
        config.NET.DINO.IS_BN_IN_HEAD = bool_flag(args.net_dino_is_bn_in_head)
    if _check_args('net_dino_num_head_layers'):
        config.NET.DINO.NUM_HEAD_LAYERS = args.net_dino_num_head_layers
    if _check_args('net_dino_head_hidden_dim'):
        config.NET.DINO.HEAD_HIDDEN_DIM = args.net_dino_head_hidden_dim
    if _check_args('net_dino_head_bottleneck_dim'):
        config.NET.DINO.HEAD_BOTTLENECK_DIM = args.net_dino_head_bottleneck_dim
    if _check_args('net_dino_head_out_dim'):
        config.NET.DINO.HEAD_OUT_DIM = args.net_dino_head_out_dim
    if _check_args('net_dino_centering'):
        config.NET.DINO.CENTERING = args.net_dino_centering

    if _check_args('net_pera_s_ratio'):
        config.NET.PERA.S_RATIO = args.net_pera_s_ratio
    if _check_args('net_pera_t_ratio'):
        config.NET.PERA.T_RATIO = args.net_pera_t_ratio
    if _check_args('net_pera_dino_loss_weight'):
        config.NET.PERA.DINO_LOSS_WEIGHT = args.net_pera_dino_loss_weight
    if _check_args('net_pera_ibot_loss_weight'):
        config.NET.PERA.IBOT_LOSS_WEIGHT = args.net_pera_ibot_loss_weight
    if _check_args('net_pera_koleo_loss_weight'):
        config.NET.PERA.KOLEO_LOSS_WEIGHT = args.net_pera_koleo_loss_weight
    if _check_args('net_pera_mae_loss_weight'):
        config.NET.PERA.MAE_LOSS_WEIGHT = args.net_pera_mae_loss_weight



    
    # loss setting
    if _check_args('loss_name'):
        config.LOSS.NAME = none_flag(config.NET.NAME.lower())
    if _check_args('loss_is_average'):
        config.LOSS.IS_AVERAGE = bool_flag(args.loss_is_average)


    # optimizer setting
    if _check_args('optim_name'):
        config.OPTIMIZER.NAME = none_flag(args.optim_name.lower())
    if _check_args('optim_momentum'):
        config.OPTIMIZER.MOMENTUM = args.optim_momentum
    if _check_args('optim_eps'):
        config.OPTIMIZER.EPS = args.optim_eps
    if _check_args('optim_betas'):
        config.OPTIMIZER.BETAS = args.optim_betas

    # lr scheduler setting
    if _check_args('learning_rate'):
        config.LR_SCHEDULER.LEARNING_RATE = args.learning_rate
    if _check_args('lrs_final_value'):
        config.LR_SCHEDULER.FINAL_VALUE = args.lrs_final_value
    if _check_args('lrs_warmup_epochs'):
        config.LR_SCHEDULER.WARMUP_EPOCHS = args.lrs_warmup_epochs
    if _check_args('lrs_warmup_value'):
        config.LR_SCHEDULER.WARMUP_VALUE = args.lrs_warmup_value
    if _check_args('lrs_freeze_epochs'):
        config.LR_SCHEDULER.FREEZE_EPOCHS = args.lrs_freeze_epochs
    if _check_args('lrs_is_restart'):
        config.LR_SCHEDULER.IS_RESTART = bool_flag(args.lrs_is_restart)
    if _check_args('lrs_T_0'):
        config.LR_SCHEDULER.T_0 = args.lrs_T_0
    if _check_args('lrs_T_mult'):
        config.LR_SCHEDULER.T_MULT = args.lrs_T_mult
    
    # weight decay scheduler setting
    if _check_args('weight_decay'):
        config.WD_SCHEDULER.WEIGHT_DECAY = args.weight_decay
    if _check_args('wds_final_value'):
        config.WD_SCHEDULER.FINAL_VALUE = args.wds_final_value
    if _check_args('wds_warmup_epochs'):
        config.WD_SCHEDULER.WARMUP_EPOCHS = args.wds_warmup_epochs
    if _check_args('wds_warmup_value'):
        config.WD_SCHEDULER.WARMUP_VALUE = args.wds_warmup_value
    if _check_args('wds_freeze_epochs'):
        config.WD_SCHEDULER.FREEZE_EPOCHS = args.wds_freeze_epochs
    if _check_args('wds_is_restart'):
        config.WD_SCHEDULER.IS_RESTART = bool_flag(args.wds_is_restart)
    if _check_args('wds_T_0'):
        config.WD_SCHEDULER.T_0 = args.wds_T_0
    if _check_args('wds_T_mult'):
        config.WD_SCHEDULER.T_MULT = args.wds_T_mult

    # teacher momentum scheduler setting
    if _check_args('teacher_momentum'):
        config.TM_SCHEDULER.TEACHER_MOMENTUM = args.teacher_momentum
    if _check_args('tms_final_value'):
        config.TM_SCHEDULER.FINAL_VALUE = args.tms_final_value
    if _check_args('tms_warmup_epochs'):
        config.TM_SCHEDULER.WARMUP_EPOCHS = args.tms_warmup_epochs
    if _check_args('tms_warmup_value'):
        config.TM_SCHEDULER.WARMUP_VALUE = args.tms_warmup_value
    if _check_args('tms_freeze_epochs'):
        config.TM_SCHEDULER.FREEZE_EPOCHS = args.tms_freeze_epochs
    if _check_args('tms_is_restart'):
        config.TM_SCHEDULER.IS_RESTART = bool_flag(args.tms_is_restart)
    if _check_args('tms_T_0'):
        config.TM_SCHEDULER.T_0 = args.tms_T_0
    if _check_args('tms_T_mult'):
        config.TM_SCHEDULER.T_MULT = args.tms_T_mult

    # student temperature setting
    if _check_args('student_temp'):
        config.STUDENT_TEMP = args.student_temp
    # teacher temperature scheduler setting
    if _check_args('teacher_temp'):
        config.TT_SCHEDULER.TEACHER_TEMP = args.teacher_temp
    if _check_args('tts_final_value'):
        config.TT_SCHEDULER.FINAL_VALUE = args.tts_final_value
    if _check_args('tts_warmup_epochs'):
        config.TT_SCHEDULER.WARMUP_EPOCHS = args.tts_warmup_epochs
    if _check_args('tts_warmup_value'):
        config.TT_SCHEDULER.WARMUP_VALUE = args.tts_warmup_value
    if _check_args('tts_freeze_epochs'):
        config.TT_SCHEDULER.FREEZE_EPOCHS = args.tts_freeze_epochs
    if _check_args('tts_is_restart'):
        config.TT_SCHEDULER.IS_RESTART = bool_flag(args.tts_is_restart)
    if _check_args('tts_T_0'):
        config.TT_SCHEDULER.T_0 = args.tts_T_0
    if _check_args('tts_T_mult'):
        config.TT_SCHEDULER.T_MULT = args.tts_T_mult
    
    if _check_args('dtype'):
        config.DTYPE = none_flag(args.dtype)
    if _check_args('gradient_clipping'):
        config.GRADIENT_CLIPPING = none_flag(args.gradient_clipping)
    if _check_args('gradient_accumulation_steps'):
        config.GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
        
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = base_cfg.clone()
    if args:
        update_config(config, args)
    return config


def save_config(config):
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    path = f"{config.OUTPUT_PATH}/{config.NET.NAME}_{config.CFG_NOTE}_{time.strftime('%y%m%d%H%M%S')}.yaml"
    with open(path, "w") as f:
        f.write(config.dump())

def get_output_path(cfg, mode='Pretrain'):
    dataset = (cfg.DATA.PRETRAIN_DATA_PATH).split('/')[-1]
    
    num_resume = 0
    if cfg.IS_RESUME:
        src_output_path = ('/'.join(cfg.OUTPUT_PATH.split('/')[:-1]))
        num_resume = len([w for w in os.listdir(src_output_path) if os.path.isdir(os.path.join(src_output_path, w))])
        if num_resume == 0:
            print("No previous checkpoints found. Fail to resume training.")
            sys.exit(1)
        else:
            print(f"Found {num_resume} previous restart checkpoints. This is the {num_resume + 1}th restart, restart epoch: {cfg.START_EPOCH}.")
            output_path = os.path.join(src_output_path, f"Run{num_resume + 1}")
    else:
        output_path = os.path.join(cfg.OUTPUT_PATH, mode, cfg.NET.NAME, dataset, 
                                f"{cfg.CFG_NOTE}_{time.strftime('%y%m%d%H%M%S')}", f"Run{num_resume + 1}")
        if os.path.exists(output_path):
            print(f"Output path {output_path} already exists, please wait for a moment.")
            sys.exit(1)
    return output_path


def check_config(cfg, logger):
    is_exit = False

    # Pretrain config check
    if cfg.CFG_PATH is None:
        logger.warning("The config file path is empty. Please verify that the pre-training uses the default config.")

    if cfg.PRETRAIN_PATH is None:
        if cfg.IS_RESUME:
            logger.warning("Resume training mode, but the pre-trained model path is empty. Please verify the pre-trained model path. The program will exit.")
            is_exit = True
        else:
            logger.warning("The pre-trained model path is empty; training will start from scratch.")
    else:
        if len(os.path.split(cfg.PRETRAIN_PATH)[-1].split('_')) == 4:
            if cfg.IS_RESUME:
                logger.warning("Resume training mode and continue training using the pre-trained model.")
            else:
                logger.warning("Resume training mode was not used, training will start from scratch.")
            

    if cfg.OUTPUT_PATH is None:
        logger.warning("The output path is empty. Please set an output path, the program will exit.")
        is_exit = True
    else:
        if cfg.IS_RESUME:
            logger.warning("Resume training mode, and the output path will be overwritten with the output path from the last training.")


    if not os.path.exists(cfg.DATA.PRETRAIN_DATA_PATH):
        logger.warning("The pre-training dataset path does not exist. The program will exit.")
        is_exit = True


    if cfg.NET.NAME is None:
        logger.warning("The network name is empty. Please set a network name; otherwise, the program will exit.")
        is_exit = True

    if cfg.LOSS.NAME is None:
        logger.warning("The loss function name is empty. Please set the loss function name. The program will exit.")
        is_exit = True

    if cfg.OPTIMIZER.NAME is None:
        logger.warning("The optimizer name is empty. Please set the optimizer name; otherwise, the program will exit.")
        is_exit = True

    if is_exit:
        sys.exit(1)