import torch
from torch import nn
from torch.nn import functional as F


class RBert_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        print(' input_dim: %d \n output_dim: %d'
              % (input_dim, output_dim))
        super(RBert_Model, self).__init__()

        self.sen_li = nn.Linear(input_dim, 512)  # sentence vector linear layer
        self.sen_bn = nn.BatchNorm1d(512)
        self.f_ent_li = nn.Linear(input_dim, 512)  # first entity linear layer
        self.f_ent_bn = nn.BatchNorm1d(512)
        self.s_ent_li = nn.Linear(input_dim, 512)  # second entity linear layer
        self.s_ent_bn = nn.BatchNorm1d(512)

        self.hidden2 = nn.Linear(512 * 3, 100)
        self.bn2 = nn.BatchNorm1d(100)
        # self.hidden3 = nn.Linear(256, 100)
        # self.bn3 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x, weights=None):
        sentence_vec = x[:, :1, :].view(-1, 1024)
        first_ent_vec = x[:, 1:2, :].view(-1, 1024)
        second_ent_vec = x[:, 2:, :].view(-1, 1024)
        sentence_vec = F.relu(self.sen_bn(self.sen_li(sentence_vec)))
        first_ent_vec = F.relu(self.f_ent_bn(self.f_ent_li(first_ent_vec)))
        second_ent_vec = F.relu(self.s_ent_bn(self.s_ent_li(second_ent_vec)))

        x = torch.cat((sentence_vec, first_ent_vec, second_ent_vec), dim=1)
        x = F.relu(self.bn2(self.hidden2(x)))
        # x = F.relu(self.bn3(self.hidden3(x)))
        x = self.out(x)
        return x

    # def accuracy(self, predictions, targets):
    #     predictions = predictions.argmax(dim=1).view(targets.shape)
    #     return (predictions == targets).sum().float() / targets.size(0)

    # def calc_loss(self, x, y, net):
    #     # calc loss of net(x) and y, return loss and the correct
    #     logits = net(x)
    #     loss = F.cross_entropy(logits, y)
    #
    #     pred = F.softmax(logits, dim=1).argmax(dim=1)
    #     correct = torch.eq(pred, y).sum().item()
    #     return loss, correct

    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def calc_loss(self, x, y, net, rel_emb_h=None, rel_emb_t=None, rel_emb_ht=None):
        # calc loss of net(x) and y, return loss and the correct
        logits = net(x)
        loss = F.cross_entropy(logits, y)

        pred = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred, y).sum().item()
        return loss, correct
