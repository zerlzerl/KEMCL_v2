from data_prepare import BaseLoader
import os
import json
import random
from bert_serving.client import BertClient
from tqdm import tqdm
import pickle as pkl
import numpy as np
from transformers import BertTokenizer, BertModel


class DataLoader(BaseLoader):

    def init(self):
        self.bert_client = None
        # override BaseLoader's init method
        self.load_benchmark()
        # load extra file
        self.load_extra()

        # construct training tasks
        print('Construct training tasks data')
        self.construct_training_tasks()

        # construct testing tasks
        print('Construct testing tasks data')
        self.construct_testing_tasks()

    def load_benchmark(self):
        self.all_relations = self.read_json(os.path.join(self.dataset, 'all_relations.json'))
        print('All relations number: %d' % len(self.all_relations))

        self.train_relations = self.read_json(os.path.join(self.dataset, 'train_relations.json'))

        print('Train relations number: %d' % len(self.train_relations))

        self.test_relations = self.read_json(os.path.join(self.dataset, 'test_relations.json'))
        print('Test relations number: %d' % len(self.test_relations))

        self.train_data = self.read_json(os.path.join(self.dataset, 'train.json'))
        train_total = 0
        for rel in self.train_data:
            train_total += len(self.train_data[rel])
        print('Train items number: %d' % train_total)

        if self.sentence_encode_mode == 'full':
            train_vec_file = 'full_train_vec.pkl'
        elif self.sentence_encode_mode == 'words_between':
            train_vec_file = 'words_between_train_vec.pkl'
        elif self.sentence_encode_mode == 'full_with_ent':
            train_vec_file = 'full_with_ent_train_vec.pkl'


        if os.path.exists(os.path.join(self.dataset, train_vec_file)):
            print('Load train embeddings')
            self.train_vec = self.read_pickle(os.path.join(self.dataset, train_vec_file))
        else:
            print('Generate train embeddings')
            if self.bert_client is None:
                self.init_bert_client()
            self.train_vec = self.generate_embeddings(self.train_data)
            print('Saving  train embeddings')
            self.dump_pickle(os.path.join(self.dataset, train_vec_file), self.train_vec)

        self.test_data = self.read_json(os.path.join(self.dataset, 'test.json'))
        test_total = 0
        for rel in self.test_data:
            test_total += len(self.test_data[rel])
        print('Test items number: %d' % test_total)

        if self.sentence_encode_mode == 'full':
            test_vec_file = 'full_test_vec.pkl'
        elif self.sentence_encode_mode == 'words_between':
            test_vec_file = 'words_between_test_vec.pkl'
        elif self.sentence_encode_mode == 'full_with_ent':
            test_vec_file = 'full_with_ent_test_vec.pkl'

        if os.path.exists(os.path.join(self.dataset, test_vec_file)):
            print('Load test embeddings')
            self.test_vec = self.read_pickle(os.path.join(self.dataset, test_vec_file))
        else:
            print('Generate test embeddings')
            if self.bert_client is None:
                self.init_bert_client()
            self.test_vec = self.generate_embeddings(self.test_data)
            print('Saving  test embeddings')
            self.dump_pickle(os.path.join(self.dataset, test_vec_file), self.test_vec)

    def generate_embeddings(self, rel2items_dict, view=None):
        rel2vec_dict = {}
        for relation_id, train_items in tqdm(rel2items_dict.items()):
            tokens_list = []
            for item in train_items:
                token = item['token']
                h = item['h']
                t = item['t']
                if self.sentence_encode_mode == 'full':
                    tokens_list.append(self.remove_invalid_token(token))
                elif self.sentence_encode_mode == 'words_between':
                    h_pos = h['pos']
                    t_pos = t['pos']
                    if h_pos[0] < t_pos[0]:
                        words_between = token[h_pos[0]:t_pos[-1]]
                    else:
                        words_between = token[t_pos[0]:h_pos[-1]]
                    tokens_list.append(self.remove_invalid_token(words_between))
            embed_vec = self.bert_client.encode(tokens_list, is_tokenized=True)
            rel2vec_dict[relation_id] = embed_vec
        return rel2vec_dict

    def construct_training_tasks(self):
        # construct training tasks with the data loader config
        # this method can be override to implement different strategies
        self.meta_train_tasks = []  # separate into 2 parts: train and test
        for epoch in tqdm(range(self.epoch)):
            # construct an epoch tasks
            epoch_tasks = []
            for task_id in range(self.task_num):
                # construct a task
                epoch_tasks.append(self.construct_task(self.train_relations, self.train_vec))
            self.meta_train_tasks.append(epoch_tasks)

    def construct_testing_tasks(self):
        # construct test_task_num epoch tasks for test
        self.meta_test_tasks = []
        for test_epoch in tqdm(range(self.test_epoch)):
            # construct an epoch test tasks
            test_epoch_tasks = []
            for task_id in range(self.task_num):
                # construct an test epoch
                test_epoch_tasks.append(self.construct_task(self.test_relations, self.test_vec))
            self.meta_test_tasks.append(test_epoch_tasks)

    def construct_task(self, relation_dict, data_vec):
        task_relations = random.sample(list(relation_dict), self.n_way)
        task_relations_details = []
        _spt_x = []  # support set x (sentence token format)
        _spt_y = []  # support set y
        _qry_x = []
        _qry_y = []
        for relation_id in task_relations:
            # relation id 2 category dict
            label = list.index(task_relations, relation_id)  # 0 to n_way - 1

            relation = relation_dict[relation_id]  # relation details
            task_relations_details.append(relation)
            relation_items_vec = data_vec[relation_id]  # relation items' vec
            random_sampled_vec_idx = np.random.choice(relation_items_vec.shape[0], self.k_spt + self.k_qry)
            random_sampled_vec = relation_items_vec[random_sampled_vec_idx]  # (20, 1024)

            # vec
            _spt_x.append(random_sampled_vec[:self.k_spt])
            _qry_x.append(random_sampled_vec[self.k_spt:])

            # label
            _spt_y.extend([label] * self.k_spt)
            _qry_y.extend([label] * self.k_qry)

        spt_x, spt_y = self.shuffle(np.concatenate(_spt_x, axis=0), np.array(_spt_y, dtype=np.long))
        qry_x, qry_y = self.shuffle(np.concatenate(_qry_x, axis=0), np.array(_qry_y, dtype=np.long))

        task = {'spt_x': spt_x, 'spt_y': spt_y, 'qry_x': qry_x, 'qry_y': qry_y, 'task_relations': task_relations_details}
        return task

    def shuffle(self, x, y):
        index = np.arange(x.shape[0])
        np.random.shuffle(index)
        return x[index], y[index]

    def init_bert_client(self):
        print('Init bert client')
        # init bert client
        if self.sentence_encode_mode == 'full_with_ent' or self.sentence_encode_mode == 'multi_view':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.bert_model = BertModel.from_pretrained('bert-large-uncased')
            self.bert_model.eval()
            self.bert_model.to(self.device)
        else:
            self.bert_client = BertClient(ip=self.bert_service_ip)
    def read_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            vec = pkl.load(f)
            for rel_id, rel_vec in vec.items():
                if np.isnan(rel_vec).any():
                    print('==%s' % rel_id)
            return vec

    def dump_pickle(self, file_path, obj):
        with open(file_path, 'wb') as f:
            return pkl.dump(obj, f)

    def read_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def next_train(self):
        if self.train_index == self.epoch:
            raise Exception('Train epoch overflow')
        task = self.meta_train_tasks[self.train_index]
        self.train_index += 1
        return task

    def next_test(self):
        return self.meta_test_tasks

    def load_extra(self):
        pass

