import torch
import torch.nn as nn
import torch.distributed


class MoCoV3Loss(nn.Module):
    def __init__(self):
        super(MoCoV3Loss, self).__init__()


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    
    def compute_loss(self, q, k, T=1.0):
                # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = self.concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * T)
    
    def forward(self, Q, K, T=1.0):
        q1, q2 = Q
        k1, k2 = K
        loss1 = self.compute_loss(q1, k2, T)
        loss2 = self.compute_loss(q2, k1, T)
        return loss1 + loss2