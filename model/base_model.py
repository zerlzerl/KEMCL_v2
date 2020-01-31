import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        print(' input_dim: %d \n output_dim: %d'
              % (input_dim, output_dim))
        super(BaseModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.hidden2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.hidden3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.hidden4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x, weights=None):
        x = F.relu(self.bn1(self.hidden1(x)))
        x = F.relu(self.bn2(self.hidden2(x)))
        x = F.relu(self.bn3(self.hidden3(x)))
        x = F.relu(self.bn4(self.hidden4(x)))
        x = self.out(x)
        return x

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

