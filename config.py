import math
import time

import torch
import numpy as np
from torch.nn import functional as F
from model import MAML
import random
from tqdm import tqdm


class Config:
    def __init__(self):
        # dataset
        self.dataset = None
        self.embedding_path = None
        # meta training and testing hyper-parameters setting
        self.epoch = 1000
        self.epoch_split = 50
        self.test_epoch = 100
        self.n_way = 10
        self.k_spt = 5
        self.k_qry = 15
        self.task_num = 32
        self.meta_lr = 1e-3
        self.update_lr = 0.4
        self.update_step = 5
        self.update_step_test = 10
        self.random_seed = 1
        self.test_step = 50
        self.opt_method = None
        self.meta_optimizer = None
        # self.loss = None
        # self.loss_function = None

        # model export and import setting
        self.meta_export_dir = None
        self.meta_import_dir = None
        # tmp retain, not sure to use
        self.model_export_dir = None
        self.model_import_dir = None

        # device
        self.device_type = 'cpu'
        self.device = None

        # dependencies
        self.data_loader = None
        # inner train model
        self.model = None
        self.model_config = None

        # bert service ip
        self.bert_service_ip = '127.0.0.1'

        # sort tasks by what view
        self.difficulty_view = None
        self.sentence_encode_mode = 'full'  # 'words_between'

    def init(self):
        # init dependencies
        if self.model is None:
            raise Exception('Train model not set.')

        if self.dataset is None:
            raise Exception('Dataset not set.')

        if torch.cuda.is_available() and 'cuda' in self.device_type:
            self.device = torch.device(self.device_type)
        elif self.device_type is 'cpu':
            self.device = torch.device('cpu')
        else:
            raise Exception('Device set error.')

        # set random seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # init data loader
        self.format_log('Data Config')
        self.data_config = self.get_data_config()
        self.format_log('Init data loader')
        self.dataloader = self.data_loader(self.data_config)

        # init train model
        self.format_log('Init inner train model')
        self.train_model = self.model(**self.model_config).to(self.device)

        # init meta learner
        self.format_log('Init meta learner')
        self.meta_learner = MAML(self.train_model, self.update_lr, first_order=False).to(self.device)

        if self.opt_method is None:
            raise Exception('Optimization Method not set.')
        if self.opt_method is 'Adam':
            self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=self.meta_lr)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch

    def set_n_way(self, n_way):
        self.n_way = n_way

    def set_k_spt(self, k_spt):
        self.k_spt = k_spt

    def set_k_qry(self, k_qry):
        self.k_qry = k_qry

    def set_task_num(self, task_num):
        self.task_num = task_num

    def set_meta_lr(self, meta_lr):
        self.meta_lr = meta_lr

    def set_update_lr(self, update_lr):
        self.update_lr = update_lr

    def set_update_step(self, update_step):
        self.update_step = update_step

    def set_update_step_test(self, update_step_test):
        self.update_step_test = update_step_test

    def set_test_step(self, test_step):
        self.test_step = test_step

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_meta_export_dir(self, meta_export_dir):
        self.meta_export_dir = meta_export_dir

    def set_meta_import_dir(self, meta_import_dir):
        self.meta_import_dir = meta_import_dir

    def set_model_export_dir(self, model_export_dir):
        self.model_export_dir = model_export_dir

    def set_model_import_dir(self, model_import_dir):
        self.model_import_dir = model_import_dir

    def set_model(self, model):
        self.model = model

    def set_device_type(self, device_type):
        self.device_type = device_type

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_model_config(self, **model_config):
        self.model_config = model_config

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_bert_service_ip(self, bert_service_ip):
        self.bert_service_ip = bert_service_ip

    def set_difficulty_view(self, difficulty_view):
        self.difficulty_view = difficulty_view

    def set_sentence_encode_mode(self, sentence_encode_mode):
        self.sentence_encode_mode = sentence_encode_mode

    def get_data_config(self):
        # pack the parameters which used in data loader
        print(' dataset: %s \n epoch: %d \n n_way: %d \n k_spt: %d \n k_qry: %d \n task_num: %d \n bert_service_ip: %s'
              % (self.dataset, self.epoch, self.n_way, self.k_spt, self.k_qry, self.task_num, self.bert_service_ip))
        data_config = {'dataset': self.dataset, 'epoch': self.epoch, 'epoch_split': self.epoch_split, 'n_way': self.n_way,
                       'k_spt': self.k_spt, 'k_qry': self.k_qry, 'task_num': self.task_num,
                       'bert_service_ip': self.bert_service_ip, 'test_epoch': self.test_epoch,
                       'difficulty_view': self.difficulty_view, 'sentence_encode_mode': self.sentence_encode_mode,
                       'device': self.device}
        return data_config

    def run(self):
        self.format_log('Start meta training')
        print('|Epoch\t|%s Total Time\t|' % (''.join(['Step %d\t\t\t\t\t|' % step for step in range(self.update_step + 1)])))
        start_time = time.time()
        for epoch in range(self.epoch):  # start an epoch
            qry_losses = [0 for _ in range(self.update_step + 1)]
            qry_corrects = [0 for _ in range(self.update_step + 1)]
            meta_train_task_batch = self.dataloader.next_train()  # task_num tasks

            for train_task in meta_train_task_batch:
                # unpack a train task
                spt_x = torch.from_numpy(train_task['spt_x']).to(self.device)
                spt_y = torch.from_numpy(train_task['spt_y']).to(self.device)
                qry_x = torch.from_numpy(train_task['qry_x']).to(self.device)
                qry_y = torch.from_numpy(train_task['qry_y']).to(self.device)
                if self.sentence_encode_mode == 'multi_view':
                    rel_emb_h = torch.from_numpy(train_task['rel_emb_h']).to(self.device)
                    rel_emb_t = torch.from_numpy(train_task['rel_emb_t']).to(self.device)
                    rel_emb_ht = torch.from_numpy(train_task['rel_emb_ht']).to(self.device)
                else:
                    rel_emb_h = None
                    rel_emb_t = None
                    rel_emb_ht = None
                # 1. use original meta learner run the i-th task and compute loss for k=0
                with torch.no_grad():
                    loss_qry, correct = self.train_model.calc_loss(qry_x, qry_y, self.meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                    qry_losses[0] += loss_qry
                    qry_corrects[0] += correct

                # 2. clone a independent maml learner for each task
                meta_learner = self.meta_learner.clone()

                # 3. run the i-th task and compute loss for k=1~K
                for k in range(1, self.update_step + 1):
                    loss_spt, _ = self.train_model.calc_loss(spt_x, spt_y, meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                    meta_learner.adapt(loss_spt)

                    loss_qry, correct = self.train_model.calc_loss(qry_x, qry_y, meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                    qry_losses[k] += loss_qry
                    qry_corrects[k] += correct

            # 4. end of all tasks, sum overall losses on query set across all tasks
            loss_q = qry_losses[-1] / self.task_num
            self.meta_optimizer.zero_grad()  # zero grad of meta optimizer
            # optimize theta parameters
            loss_q.backward()
            self.meta_optimizer.step()

            accs = np.array(qry_corrects) / (self.k_qry * self.n_way * self.task_num)
            # print train metrics
            now_time = time.time()
            print(('|%0' + str(math.ceil(math.log(self.epoch, 10))) + 'd\t|%s %.1fs\t\t\t|') %
            (epoch, ''.join('L: %.3f Acc: %.3f\t|' % (loss.item(), acc * 100) for loss, acc in zip(qry_losses, accs)), now_time - start_time))

            # test the model
            if (epoch + 1) % self.test_step == 0:
                self.format_log('Test After %d Epoch Meta Training' % (epoch + 1))
                test_accs = []
                meta_test_tasks = self.dataloader.next_test()
                for test_epoch_tasks in meta_test_tasks:
                    for test_task in test_epoch_tasks:
                        qry_corrects = [0 for _ in range(self.update_step_test + 1)]
                        # unpack a meta test task
                        spt_x = torch.from_numpy(test_task['spt_x']).to(self.device)
                        spt_y = torch.from_numpy(test_task['spt_y']).to(self.device)
                        qry_x = torch.from_numpy(test_task['qry_x']).to(self.device)
                        qry_y = torch.from_numpy(test_task['qry_y']).to(self.device)
                        if self.sentence_encode_mode == 'multi_view':
                            rel_emb_h = torch.from_numpy(test_task['rel_emb_h']).to(self.device)
                            rel_emb_t = torch.from_numpy(test_task['rel_emb_t']).to(self.device)
                            rel_emb_ht = torch.from_numpy(test_task['rel_emb_ht']).to(self.device)
                        else:
                            rel_emb_h = None
                            rel_emb_t = None
                            rel_emb_ht = None
                        # use original meta learner to compute loss and accuracy for k = 0
                        with torch.no_grad():
                            loss_qry, correct = self.train_model.calc_loss(qry_x, qry_y, self.meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                            qry_losses[0] += loss_qry
                            qry_corrects[0] += correct

                        # clone a independent meta learner for each task
                        meta_learner = self.meta_learner.clone()

                        # use cloned meta leaner run the i-th task and compute loss for k=1~K
                        for k in range(1, self.update_step_test + 1):
                            loss_spt, _ = self.train_model.calc_loss(spt_x, spt_y, meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                            meta_learner.adapt(loss_spt)

                            # calc loss and correct in meta test query set
                            _, correct = self.train_model.calc_loss(qry_x, qry_y, meta_learner, rel_emb_h, rel_emb_t, rel_emb_ht)
                            qry_corrects[k] += correct

                        acc = np.array(qry_corrects) / (self.n_way * self.k_qry)
                        test_accs.append(acc)

                test_accs = np.array(test_accs).mean(axis=0).astype(np.float16)
                print('Test acc: %s' % ''.join('Step%d: %.3f\t|' % (i, test_accs[i] * 100) for i in range(len(test_accs))))
                self.format_log('Continue Meta Training')

    def format_log(self, str, total_len=80, seg='='):
        str_len = len(str)
        start_num = (total_len - str_len - 2) // 2
        end_num = total_len - str_len - start_num - 2
        print('%s %s %s' % (seg * start_num, str, seg * end_num))
