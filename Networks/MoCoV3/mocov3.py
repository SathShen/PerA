import torch
import torch.nn as nn
from functools import partial
from .vit_moco import VisionTransformerMoCo


class MoCoV3(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, cfg):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCoV3, self).__init__()

        # build encoders
        self.student = VisionTransformerMoCo(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO,
                                                      qkv_bias=True,
                                                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                      drop_path_rate=cfg.NET.DROP_PATH_RATE)
        
        self.teacher = VisionTransformerMoCo(img_size=cfg.AUG.GLOBAL_CROP_SIZE,
                                                      patch_size=cfg.NET.PATCH_SIZE, 
                                                      embed_dim=cfg.NET.EMBED_DIM,
                                                      depth=cfg.NET.DEPTH,
                                                      num_heads=cfg.NET.NUM_HEADS,
                                                      mlp_ratio=cfg.NET.MLP_RATIO,
                                                      qkv_bias=True,
                                                      norm_layer=partial(nn.LayerNorm, eps=1e-6))

        del self.student.head, self.teacher.head # remove original fc layer

        # projectors
        self.student.head = self._build_mlp(3, cfg.NET.EMBED_DIM, cfg.NET.MOCO.HEAD_HIDDEN_DIM, cfg.NET.MOCO.HEAD_OUT_DIM)
        self.teacher.head = self._build_mlp(3, cfg.NET.EMBED_DIM, cfg.NET.MOCO.HEAD_HIDDEN_DIM, cfg.NET.MOCO.HEAD_OUT_DIM)

        # predictor
        self.predictor = self._build_mlp(2, cfg.NET.MOCO.HEAD_OUT_DIM, cfg.NET.MOCO.HEAD_HIDDEN_DIM, cfg.NET.MOCO.HEAD_OUT_DIM)


        for param_b, param_m in zip(self.student.parameters(), self.teacher.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    def forward(self, X):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        x1, x2 = X

        # compute features
        q1 = self.predictor(self.student(x1))
        q2 = self.predictor(self.student(x2))

        with torch.no_grad():  
            # compute momentum features as targets
            k1 = self.teacher(x1)
            k2 = self.teacher(x2)

        return (q1, q2), (k1, k2)
