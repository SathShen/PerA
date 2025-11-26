import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.center = self.center.to(teacher_output.device)
        self.apply_center_update()
        teacher_output = teacher_output.float()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        self.center = self.center.to(teacher_output.device)
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for i in range(len(student_output_list)):
            lsm = F.log_softmax(student_output_list[i].float() / self.student_temp, dim=-1)
            loss = torch.sum(teacher_out_softmaxed_centered_list[i].float() * lsm, dim=-1)
            total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output) # num_crops
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)  # (num_crops , batch_size * world_size, out_dim) -> (1, batch_size * world_size, out_dim)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True) # (1, batch_size * world_size, out_dim) all world sum

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss


class DistillLoss(nn.Module):
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
        s_global_cls_tokens = student_output['global_cls_tokens']
        s_global_pred_tokens = student_output['global_pred_tokens']
        t_global_pred_tokens = teacher_output['global_pred_tokens']

        loss_dict = {} # for display
        loss_accumulator = 0  # for backprop

        t_cls_tokens, t_patch_tokens = self.teacher_centering(teacher_output, teacher_temp)


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
        
        koleo_dino_global_loss = sum(self.koleo_loss(p) for p in s_global_cls_tokens.chunk(self.num_global))
        loss_dict["koleo_loss"] = koleo_dino_global_loss / (self.global_loss_terms)
        loss_accumulator += self.koleo_loss_weight * koleo_dino_global_loss

        mae_loss = self.patch_loss(s_global_pred_tokens, t_global_pred_tokens)
        loss_dict["mae_loss"] = mae_loss / (self.global_loss_terms * 100)
        loss_accumulator += self.mae_loss_weight * mae_loss


        return loss_accumulator, loss_dict


    def teacher_centering(self, teacher_output, teacher_temp):
        t_local_cls_tokens = teacher_output['local_cls_tokens']
        t_global_cls_tokens = teacher_output['global_cls_tokens']

        teacher_cls_tokens = torch.cat([t_global_cls_tokens, t_local_cls_tokens])    # (num_crops * bs, dim)

        if self.centering == "centering":
            teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                teacher_cls_tokens, teacher_temp=teacher_temp
                ).view(self.num_global + self.num_local, -1, *teacher_cls_tokens.shape[1:])   # reshape to (num_crops, bs, dim)
            self.dino_loss.update_center(teacher_cls_tokens)

        else:
            raise NotImplementedError
        return teacher_dino_softmaxed_centered_list, None