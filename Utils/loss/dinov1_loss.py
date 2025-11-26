import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class DinoV1Loss(nn.Module):
    def __init__(self, out_dim, ncrops, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning

    def forward(self, student_output, teacher_output, teacher_temp):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        self.center = self.center.to(student_output.device)
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)  # crop tuple ([batch_size, outdim], [batch_size, outdim]...) student共有ncrops=10个输出

        # teacher centering and sharpening
        # teacher_out shape: [batch_size * 2, outdim]
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1) # teacher输出的中心化和温度调整
        teacher_out = teacher_out.detach().chunk(2)  # crop tuple ([batch_size, outdim], [batch_size, outdim]) teacher共有2个输出
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view 具体讲跳过了两个global crop相同的情况
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)