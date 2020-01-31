import torch
from torch import nn
from torch.nn import functional as F


class MultiViewModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        print(' input_dim: %d \n output_dim: %d'
              % (input_dim, output_dim))
        super(MultiViewModel, self).__init__()

        self.sen_li_h = nn.Linear(input_dim, 256)  # sentence vector linear layer for h
        self.sen_bn_h = nn.BatchNorm1d(256)
        self.f_ent_li_h = nn.Linear(input_dim, 256)  # first entity linear layer for h
        self.f_ent_bn_h = nn.BatchNorm1d(256)
        self.s_ent_li_h = nn.Linear(input_dim, 256)  # second entity linear layer for h
        self.s_ent_bn_h = nn.BatchNorm1d(256)

        self.classifier_h_input = nn.Linear(256 * 3 + output_dim * 200, 500)
        self.classifier_h_bn1 = nn.BatchNorm1d(500)
        self.classifier_h_hidden1 = nn.Linear(500, 100)
        self.classifier_h_bn2 = nn.BatchNorm1d(100)
        self.classifier_h_output = nn.Linear(100, output_dim)

        self.sen_li_t = nn.Linear(input_dim, 256)  # sentence vector linear layer for t
        self.sen_bn_t = nn.BatchNorm1d(256)
        self.f_ent_li_t = nn.Linear(input_dim, 256)  # first entity linear layer for t
        self.f_ent_bn_t = nn.BatchNorm1d(256)
        self.s_ent_li_t = nn.Linear(input_dim, 256)  # second entity linear layer for t
        self.s_ent_bn_t = nn.BatchNorm1d(256)

        self.classifier_t_input = nn.Linear(256 * 3 + output_dim * 200, 500)
        self.classifier_t_bn1 = nn.BatchNorm1d(500)
        self.classifier_t_hidden1 = nn.Linear(500, 100)
        self.classifier_t_bn2 = nn.BatchNorm1d(100)
        self.classifier_t_output = nn.Linear(100, output_dim)

        self.sen_li_ht = nn.Linear(input_dim, 256)  # sentence vector linear layer for h&t
        self.sen_bn_ht = nn.BatchNorm1d(256)
        self.f_ent_li_ht = nn.Linear(input_dim, 256)  # first entity linear layer for h&t
        self.f_ent_bn_ht = nn.BatchNorm1d(256)
        self.s_ent_li_ht = nn.Linear(input_dim, 256)  # second entity linear layer for h&t
        self.s_ent_bn_ht = nn.BatchNorm1d(256)

        self.classifier_ht_input = nn.Linear(256 * 3 + output_dim * 200, 500)
        self.classifier_ht_bn1 = nn.BatchNorm1d(500)
        self.classifier_ht_hidden1 = nn.Linear(500, 100)
        self.classifier_ht_bn2 = nn.BatchNorm1d(100)
        self.classifier_ht_output = nn.Linear(100, output_dim)

        self.classifier_ent_sen_input = nn.Linear(1024, 256)
        self.classifier_ent_sen_bn1 = nn.BatchNorm1d(256)
        self.classifier_ent_f_ent_input = nn.Linear(1024, 256)
        self.classifier_ent_f_ent_bn1 = nn.BatchNorm1d(256)
        self.classifier_ent_s_ent_input = nn.Linear(1024, 256)
        self.classifier_ent_s_ent_bn1 = nn.BatchNorm1d(256)

        self.classifier_ent_hidden1 = nn.Linear(256 * 3, 100)
        self.classifier_ent_bn2 = nn.BatchNorm1d(100)
        self.classifier_ent_output = nn.Linear(100, output_dim)

        self.final_bn = nn.BatchNorm1d(4 * output_dim)
        self.final_output = nn.Linear(4 * output_dim, output_dim)

    def forward(self, x, rel_emb_h, rel_emb_t, rel_emb_ht, weights=None):
        sentence_vec_h = x[:, 0:1, :].view(-1, 1024)
        first_ent_vec_h = x[:, 1:2, :].view(-1, 1024)
        second_ent_vec_h = x[:, 2:3, :].view(-1, 1024)

        sentence_vec_t = x[:, 3:4, :].view(-1, 1024)
        first_ent_vec_t = x[:, 4:5, :].view(-1, 1024)
        second_ent_vec_t = x[:, 5:6, :].view(-1, 1024)

        sentence_vec_ht = x[:, 6:7, :].view(-1, 1024)
        first_ent_vec_ht = x[:, 7:8, :].view(-1, 1024)
        second_ent_vec_ht = x[:, 8:9, :].view(-1, 1024)

        sentence_vec_ent = x[:, 9:10, :].view(-1, 1024)
        first_ent_vec_ent = x[:, 10:11, :].view(-1, 1024)
        second_ent_vec_ent = x[:, 11:12, :].view(-1, 1024)

        sentence_vec_h = F.relu(self.sen_bn_h(self.sen_li_h(sentence_vec_h)))
        first_ent_vec_h = F.relu(self.f_ent_bn_h(self.f_ent_li_h(first_ent_vec_h)))
        second_ent_vec_h = F.relu(self.s_ent_bn_h(self.s_ent_li_h(second_ent_vec_h)))

        present_h = torch.cat((sentence_vec_h, first_ent_vec_h, second_ent_vec_h), dim=1)
        rel_emb_h = rel_emb_h.view(1, -1)
        rel_emb_h = rel_emb_h.expand(present_h.shape[0], rel_emb_h.shape[1])
        h_feature = torch.cat((present_h, rel_emb_h), dim=1)

        h_feature = F.relu(self.classifier_h_bn1(self.classifier_h_input(h_feature)))
        h_feature = F.relu(self.classifier_h_bn2(self.classifier_h_hidden1(h_feature)))
        h_output = self.classifier_h_output(h_feature)

        sentence_vec_t = F.relu(self.sen_bn_t(self.sen_li_t(sentence_vec_t)))
        first_ent_vec_t = F.relu(self.f_ent_bn_t(self.f_ent_li_t(first_ent_vec_t)))
        second_ent_vec_t = F.relu(self.s_ent_bn_t(self.s_ent_li_t(second_ent_vec_t)))

        present_t = torch.cat((sentence_vec_t, first_ent_vec_t, second_ent_vec_t), dim=1)
        rel_emb_t = rel_emb_t.view(1, -1)
        rel_emb_t = rel_emb_t.expand(present_t.shape[0], rel_emb_t.shape[1])
        t_feature = torch.cat((present_t, rel_emb_t), dim=1)

        t_feature = F.relu(self.classifier_t_bn1(self.classifier_t_input(t_feature)))
        t_feature = F.relu(self.classifier_t_bn2(self.classifier_t_hidden1(t_feature)))
        t_output = self.classifier_t_output(t_feature)

        sentence_vec_ht = F.relu(self.sen_bn_ht(self.sen_li_ht(sentence_vec_ht)))
        first_ent_vec_ht = F.relu(self.f_ent_bn_ht(self.f_ent_li_ht(first_ent_vec_ht)))
        second_ent_vec_ht = F.relu(self.s_ent_bn_ht(self.s_ent_li_ht(second_ent_vec_ht)))

        present_ht = torch.cat((sentence_vec_ht, first_ent_vec_ht, second_ent_vec_ht), dim=1)
        rel_emb_ht = rel_emb_ht.view(1, -1)
        rel_emb_ht = rel_emb_ht.expand(present_ht.shape[0], rel_emb_ht.shape[1])
        ht_feature = torch.cat((present_ht, rel_emb_ht), dim=1)

        ht_feature = F.relu(self.classifier_ht_bn1(self.classifier_ht_input(ht_feature)))
        ht_feature = F.relu(self.classifier_ht_bn2(self.classifier_ht_hidden1(ht_feature)))
        ht_output = self.classifier_ht_output(ht_feature)

        sentence_feature = F.relu(self.classifier_ent_sen_bn1(self.classifier_ent_sen_input(sentence_vec_ent)))
        f_ent_feature = F.relu(self.classifier_ent_f_ent_bn1(self.classifier_ent_f_ent_input(first_ent_vec_ent)))
        s_ent_feature = F.relu(self.classifier_ent_s_ent_bn1(self.classifier_ent_s_ent_input(second_ent_vec_ent)))

        ent_feature = torch.cat((sentence_feature, f_ent_feature, s_ent_feature), dim=1)
        ent_output = self.classifier_ent_output(F.relu(self.classifier_ent_bn2(self.classifier_ent_hidden1(ent_feature))))

        output = self.final_output(F.relu(self.final_bn(torch.cat((h_output, t_output, ht_output, ent_output), dim=1))))

        return output

    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def calc_loss(self, x, y, net, rel_emb_h=None, rel_emb_t=None, rel_emb_ht=None):
        # calc loss of net(x) and y, return loss and the correct
        logits = net(x, rel_emb_h, rel_emb_t, rel_emb_ht)
        loss = F.cross_entropy(logits, y)

        pred = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred, y).sum().item()
        return loss, correct

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
