import torch
from torch import nn
from torch.nn import functional as F


class MetaLeaner(nn.Module):
    def __init__(self, net, meta_lr, update_lr, update_step, update_step_test, opt_method, loss):
        super(MetaLeaner, self).__init__()
        self.net = net
        self.meta_lr = meta_lr
        self.update_lr = update_lr
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.opt_method = opt_method
        self.loss = loss

        # init loss function
        if self.loss is 'MSE':
            self.loss_function = F.mse_loss
        elif self.loss is 'cross_entropy':
            self.loss_function = F.cross_entropy

        # init optimizer
        if self.opt_method is 'Adam':
            self.meta_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, task_epoch, device):
        task_num = len(task_epoch)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            task = task_epoch[i]
            spt_x = torch.from_numpy(task['spt_x']).to(device)
            spt_y = torch.from_numpy(task['spt_y']).to(device)
            qry_x = torch.from_numpy(task['qry_x']).to(device)
            qry_y = torch.from_numpy(task['qry_y']).to(device)

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(spt_x, weights=None)
            loss = self.loss_function(logits, spt_y)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(qry_x, weights=self.net.parameters())
                loss_q = self.loss_function(logits_q, qry_y)
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, qry_y).sum().item()
                corrects[0] = corrects[0] + correct

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter