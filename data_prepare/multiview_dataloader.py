import random
import numpy as np
from data_prepare import DataLoader
import os
from tqdm import tqdm
import torch

class MultiViewDataLoader(DataLoader):
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

        self.test_data = self.read_json(os.path.join(self.dataset, 'test.json'))
        test_total = 0
        for rel in self.test_data:
            test_total += len(self.test_data[rel])
        print('Test items number: %d' % test_total)

        print('Load kl-div distance of distribution of head hypernyms')
        self.kl_dist_h = self.read_json(os.path.join(self.dataset, 'kl_dist_h.json'))

        print('Load kl-div distance of distribution of tail hypernyms')
        self.kl_dist_t = self.read_json(os.path.join(self.dataset, 'kl_dist_t.json'))

        print('Load kl-div distance of distribution of head-tail hypernyms')
        self.kl_dist_ht = self.read_json(os.path.join(self.dataset, 'kl_dist_ht.json'))

        print('Load entity to hypernyms dictionary')
        self.e2h = self.read_json(os.path.join(self.dataset, 'e2h.json'))

        print('Load relation embedding vectors related to head hypernyms')
        self.rel_vec_h = self.read_json(os.path.join(self.dataset, 'rel_vec_h.json'))

        print('Load relation embedding vectors related to tail hypernyms')
        self.rel_vec_t = self.read_json(os.path.join(self.dataset, 'rel_vec_t.json'))

        print('Load relation embedding vectors related to head and tail hypernyms')
        self.rel_vec_ht = self.read_json(os.path.join(self.dataset, 'rel_vec_ht.json'))


        # generate embeddings
        if self.sentence_encode_mode == 'multi_view':
            h_train_vec_file = 'h_train_vec.pkl'
            t_train_vec_file = 't_train_vec.pkl'
            ht_train_vec_file = 'ht_train_vec.pkl'
            ent_train_vec_file = 'ent_train_vec.pkl'

            h_test_vec_file = 'h_test_vec.pkl'
            t_test_vec_file = 't_test_vec.pkl'
            ht_test_vec_file = 'ht_test_vec.pkl'
            ent_test_vec_file = 'ent_test_vec.pkl'

        # change head to head's hypernyms
        if os.path.exists(os.path.join(self.dataset, h_train_vec_file)):
            print('Load train vectors replace h entity with h hyper')
            self.h_train_vec = self.read_pickle(os.path.join(self.dataset, h_train_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate train vectors replace h entity with h hyper')
            self.h_train_vec = self.generate_embeddings(self.train_data, 'h')
            self.dump_pickle(os.path.join(self.dataset, h_train_vec_file), self.h_train_vec)

        if os.path.exists(os.path.join(self.dataset, h_test_vec_file)):
            print('Load test vectors replace h entity with h hyper')
            self.h_test_vec = self.read_pickle(os.path.join(self.dataset, h_test_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate test vectors replace h entity with h hyper')
            self.h_test_vec = self.generate_embeddings(self.test_data, 'h')
            self.dump_pickle(os.path.join(self.dataset, h_test_vec_file), self.h_test_vec)

        # change tail to tail's hypernyms
        if os.path.exists(os.path.join(self.dataset, t_train_vec_file)):
            print('Load train vectors replace t entity with t hyper')
            self.t_train_vec = self.read_pickle(os.path.join(self.dataset, t_train_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate train vectors replace t entity with t hyper')
            self.t_train_vec = self.generate_embeddings(self.train_data, 't')
            self.dump_pickle(os.path.join(self.dataset, t_train_vec_file), self.t_train_vec)

        if os.path.exists(os.path.join(self.dataset, t_test_vec_file)):
            print('Load test vectors replace t entity with t hyper')
            self.t_test_vec = self.read_pickle(os.path.join(self.dataset, t_test_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate test vectors replace t entity with t hyper')
            self.t_test_vec = self.generate_embeddings(self.test_data, 't')
            self.dump_pickle(os.path.join(self.dataset, t_test_vec_file), self.t_test_vec)

        # change head and tail to their hypernyms
        if os.path.exists(os.path.join(self.dataset, ht_train_vec_file)):
            print('Load train vectors replace h&t entity with h&t hyper')
            self.ht_train_vec = self.read_pickle(os.path.join(self.dataset, ht_train_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate train vectors replace h&t entity with h&t hyper')
            self.ht_train_vec = self.generate_embeddings(self.train_data, 'ht')
            self.dump_pickle(os.path.join(self.dataset, ht_train_vec_file), self.ht_train_vec)

        if os.path.exists(os.path.join(self.dataset, ht_test_vec_file)):
            print('Load test vectors replace h&t entity with h&t hyper')
            self.ht_test_vec = self.read_pickle(os.path.join(self.dataset, ht_test_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate test vectors replace h&t entity with h&t hyper')
            self.ht_test_vec = self.generate_embeddings(self.test_data, 'ht')
            self.dump_pickle(os.path.join(self.dataset, ht_test_vec_file), self.ht_test_vec)

        # don't change anything
        if os.path.exists(os.path.join(self.dataset, ent_train_vec_file)):
            print('Load train vectors')
            self.ent_train_vec = self.read_pickle(os.path.join(self.dataset, ent_train_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate train vectors')
            self.ent_train_vec = self.generate_embeddings(self.train_data)
            self.dump_pickle(os.path.join(self.dataset, ent_train_vec_file), self.ent_train_vec)

        if os.path.exists(os.path.join(self.dataset, ent_test_vec_file)):
            print('Load test vectors')
            self.ent_test_vec = self.read_pickle(os.path.join(self.dataset, ent_test_vec_file))
        else:
            if self.bert_client is None:
                self.init_bert_client()
            print('Generate test vectors')
            self.ent_test_vec = self.generate_embeddings(self.test_data)
            self.dump_pickle(os.path.join(self.dataset, ent_test_vec_file), self.ent_test_vec)

    def construct_training_tasks(self):
        # construct training tasks with the data loader config
        # this method can be override to implement different strategies
        self.meta_train_tasks = []  # separate into 2 parts: train and test
        for epoch in tqdm(range(self.epoch_split)):
            # construct an epoch tasks
            epoch_tasks = []
            for task_id in range(self.task_num):
                # construct a task
                # sample a set of train relations
                task_relations = random.sample(list(self.train_relations), self.n_way)
                task_relations_details = []
                _spt_x = []  # support set x (sentence token format)
                _spt_y = []  # support set y
                _qry_x = []
                _qry_y = []
                rel_emb_h = []
                rel_emb_t = []
                rel_emb_ht = []
                # iterate evert relation
                for relation_id in task_relations:
                    label = list.index(task_relations, relation_id)  # label
                    relation = self.train_relations[relation_id]  # relation details
                    task_relations_details.append(relation)
                    # get relation items vec from different views
                    relation_items_h_vec = self.h_train_vec[relation_id]
                    relation_items_t_vec = self.t_train_vec[relation_id]
                    relation_items_ht_vec = self.ht_train_vec[relation_id]
                    relation_items_ent_vec = self.ent_train_vec[relation_id]

                    # get a random sampled vec index
                    random_sampled_vec_idx = np.random.choice(relation_items_h_vec.shape[0], self.k_spt + self.k_qry)

                    # get item's vec from different views
                    item_h_vec = relation_items_h_vec[random_sampled_vec_idx]
                    item_t_vec = relation_items_t_vec[random_sampled_vec_idx]
                    item_ht_vec = relation_items_ht_vec[random_sampled_vec_idx]
                    item_ent_vec = relation_items_ent_vec[random_sampled_vec_idx]

                    # get relations' embeddings from different view
                    relation_index = self.all_relations[relation_id]['index']
                    rel_vec_h = self.rel_vec_h[relation_index]
                    rel_vec_t = self.rel_vec_t[relation_index]
                    rel_vec_ht = self.rel_vec_ht[relation_index]

                    # split
                    spt_h_vec = item_h_vec[:self.k_spt]
                    qry_h_vec = item_h_vec[self.k_spt:]

                    spt_t_vec = item_t_vec[:self.k_spt]
                    qry_t_vec = item_t_vec[self.k_spt:]

                    spt_ht_vec = item_ht_vec[:self.k_spt]
                    qry_ht_vec = item_ht_vec[self.k_spt:]

                    spt_ent_vec = item_ent_vec[:self.k_spt]
                    qry_ent_vec = item_ent_vec[self.k_spt:]

                    spt_label = [label] * self.k_spt
                    qry_label = [label] * self.k_qry

                    spt_vec = np.concatenate((spt_h_vec, spt_t_vec, spt_ht_vec, spt_ent_vec), axis=1)
                    qry_vec = np.concatenate((qry_h_vec, qry_t_vec, qry_ht_vec, qry_ent_vec), axis=1)

                    _spt_x.append(spt_vec)
                    _spt_y.extend(spt_label)

                    _qry_x.append(qry_vec)
                    _qry_y.extend(qry_label)

                    rel_emb_h.append(rel_vec_h)
                    rel_emb_t.append(rel_vec_t)
                    rel_emb_ht.append(rel_vec_ht)

                # post process
                spt_x, spt_y = self.shuffle(np.concatenate(_spt_x, axis=0), np.array(_spt_y, dtype=np.long))
                qry_x, qry_y = self.shuffle(np.concatenate(_qry_x, axis=0), np.array(_qry_y, dtype=np.long))

                task = {'spt_x': spt_x, 'spt_y': spt_y, 'qry_x': qry_x, 'qry_y': qry_y,
                        'rel_emb_h': np.array(rel_emb_h, dtype=np.float32),
                        'rel_emb_t': np.array(rel_emb_t, dtype=np.float32),
                        'rel_emb_ht': np.array(rel_emb_ht, dtype=np.float32),
                        'task_relations': task_relations_details}

                epoch_tasks.append(task)
            self.meta_train_tasks.append(epoch_tasks)
        self.epoch -= self.epoch_split

    def construct_testing_tasks(self):
        # construct training tasks with the data loader config
        # this method can be override to implement different strategies
        self.meta_test_tasks = []  # separate into 2 parts: train and test
        for test_epoch in tqdm(range(self.test_epoch)):
            # construct an epoch test tasks
            test_epoch_tasks = []
            for task_id in range(self.task_num):
                # construct a task
                # sample a set of train relations
                task_relations = random.sample(list(self.test_relations), self.n_way)
                _spt_x = []  # support set x (sentence token format)
                _spt_y = []  # support set y
                _qry_x = []
                _qry_y = []
                rel_emb_h = []
                rel_emb_t = []
                rel_emb_ht = []
                # iterate evert relation
                for relation_id in task_relations:
                    label = list.index(task_relations, relation_id)  # label

                    # get relation items vec from different views
                    relation_items_h_vec = self.h_test_vec[relation_id]
                    relation_items_t_vec = self.t_test_vec[relation_id]
                    relation_items_ht_vec = self.ht_test_vec[relation_id]
                    relation_items_ent_vec = self.ent_test_vec[relation_id]

                    # get a random sampled vec index
                    random_sampled_vec_idx = np.random.choice(relation_items_h_vec.shape[0], self.k_spt + self.k_qry)

                    # get item's vec from different views
                    item_h_vec = relation_items_h_vec[random_sampled_vec_idx]
                    item_t_vec = relation_items_t_vec[random_sampled_vec_idx]
                    item_ht_vec = relation_items_ht_vec[random_sampled_vec_idx]
                    item_ent_vec = relation_items_ent_vec[random_sampled_vec_idx]

                    # get relations' embeddings from different view
                    relation_index = self.all_relations[relation_id]['index']
                    rel_vec_h = self.rel_vec_h[relation_index]
                    rel_vec_t = self.rel_vec_t[relation_index]
                    rel_vec_ht = self.rel_vec_ht[relation_index]


                    # split
                    spt_h_vec = item_h_vec[:self.k_spt]
                    qry_h_vec = item_h_vec[self.k_spt:]

                    spt_t_vec = item_t_vec[:self.k_spt]
                    qry_t_vec = item_t_vec[self.k_spt:]

                    spt_ht_vec = item_ht_vec[:self.k_spt]
                    qry_ht_vec = item_ht_vec[self.k_spt:]

                    spt_ent_vec = item_ent_vec[:self.k_spt]
                    qry_ent_vec = item_ent_vec[self.k_spt:]

                    spt_label = [label] * self.k_spt
                    qry_label = [label] * self.k_qry

                    spt_vec = np.concatenate((spt_h_vec, spt_t_vec, spt_ht_vec, spt_ent_vec), axis=1)
                    qry_vec = np.concatenate((qry_h_vec, qry_t_vec, qry_ht_vec, qry_ent_vec), axis=1)

                    _spt_x.append(spt_vec)
                    _spt_y.extend(spt_label)

                    _qry_x.append(qry_vec)
                    _qry_y.extend(qry_label)

                    rel_emb_h.append(rel_vec_h)
                    rel_emb_t.append(rel_vec_t)
                    rel_emb_ht.append(rel_vec_ht)

                # post process
                spt_x, spt_y = self.shuffle(np.concatenate(_spt_x, axis=0), np.array(_spt_y, dtype=np.long))
                qry_x, qry_y = self.shuffle(np.concatenate(_qry_x, axis=0), np.array(_qry_y, dtype=np.long))

                task = {'spt_x': spt_x, 'spt_y': spt_y, 'qry_x': qry_x, 'qry_y': qry_y,
                        'rel_emb_h': np.array(rel_emb_h, dtype=np.float32),
                        'rel_emb_t': np.array(rel_emb_t, dtype=np.float32),
                        'rel_emb_ht': np.array(rel_emb_ht, dtype=np.float32)}

                test_epoch_tasks.append(task)
            self.meta_test_tasks.append(test_epoch_tasks)



    def generate_embeddings(self, rel2items_dict, view=None):
        rel2vec_dict = {}
        for relation_id, train_items in tqdm(rel2items_dict.items()):
            emb_vec = torch.zeros(len(train_items), 3, 1024)
            symbol_pos = []
            for index, item in enumerate(train_items):
                item = self.replace_entity(item, view)
                token = item['token']
                h = item['h']
                t = item['t']
                token.insert(0, '[CLS]')
                if h['pos'][0] < t['pos'][0]:
                    first_ent = h
                    second_ent = t
                else:
                    first_ent = t
                    second_ent = h

                dollar_start = first_ent['pos'][0] + 1
                dollar_end = first_ent['pos'][-1] + 2
                sharp_start = second_ent['pos'][0] + 3
                sharp_end = second_ent['pos'][-1] + 4
                token.insert(dollar_start, '$')
                token.insert(dollar_end, '$')
                token.insert(sharp_start, '#')
                token.insert(sharp_end, '#')

                indexed_tokens = self.tokenizer.convert_tokens_to_ids(token)
                # tokens_list.append(indexed_tokens)
                symbol_pos.append([dollar_start, dollar_end, sharp_start, sharp_end])

                indexed_tokens = torch.tensor([indexed_tokens]).to(self.device)
                with torch.no_grad():
                    outputs = self.bert_model(indexed_tokens)
                    encoded_layers = outputs[0]
                    sentence_vec = encoded_layers[0][0]
                    first_ent_vec = torch.mean(encoded_layers[0][dollar_start + 1: dollar_end].view(-1, 1024), dim=0)
                    second_ent_vec = torch.mean(encoded_layers[0][sharp_start + 1: sharp_end].view(-1, 1024), dim=0)
                    vec = torch.cat(
                        (sentence_vec.view(-1, 1024), first_ent_vec.view(-1, 1024), second_ent_vec.view(-1, 1024)),
                        dim=0)
                    if torch.sum(torch.isnan(vec)).item() > 0:
                        print('relation: %s index: %d' % (relation_id, index))
                        emb_vec[index] = emb_vec[index - 1]
                    else:
                        emb_vec[index] = vec

            rel2vec_dict[relation_id] = emb_vec.cpu().detach().numpy()
        return rel2vec_dict

    def replace_entity(self, item, view=None):
        token = item['token']
        h = item['h']
        t = item['t']
        if view is None:
            # replace nothing
            return item
        elif view == 'h':
            # replace head entity
            h_pos = h['pos']
            h_pos_s = h_pos[0]
            h_pos_e = h_pos[-1]
            h_len = h_pos_e - h_pos_s

            if h['id'] in self.e2h:
                h_hyper = self.e2h[h['id']][0]['label']
                h_hyper_len = len(h_hyper)
            else:
                return item

            # compare and replace
            replaced_token = token[: h_pos_s] + h_hyper + token[h_pos_e:]
            replaced_h_pos_e = h_pos_e - h_len + h_hyper_len

            replaced_h = {'name': ' '.join(h_hyper), 'pos': [h_pos_s, replaced_h_pos_e]}
            # handle modification of t entity
            if replaced_h_pos_e != h_pos_e and h['pos'][-1] <= t['pos'][0]:
                t['pos'][0] += replaced_h_pos_e - h_pos_e
                t['pos'][-1] += replaced_h_pos_e - h_pos_e

            replaced_item = {'token': replaced_token, 'h': replaced_h, 't': t}
            return replaced_item
        elif view == 't':
            # replace tail entity
            t_pos = t['pos']
            t_pos_s = t_pos[0]
            t_pos_e = t_pos[-1]
            t_len = t_pos_e - t_pos_s

            if t['id'] in self.e2h:
                t_hyper = self.e2h[t['id']][0]['label']
                t_hyper_len = len(t_hyper)
            else:
                return item

            # compare and replace
            replaced_token = token[:t_pos_s] + t_hyper + token[t_pos_e:]
            replaced_t_pos_e = t_pos_e - t_len + t_hyper_len

            replaced_t = {'name': ' '.join(t_hyper), 'pos': [t_pos_s, replaced_t_pos_e]}
            # handle modification of h entity
            if replaced_t_pos_e != t_pos_e and t['pos'][-1] <= h['pos'][0]:
                h['pos'][0] += replaced_t_pos_e - t_pos_e
                h['pos'][-1] += replaced_t_pos_e - t_pos_e

            replaced_item = {'token': replaced_token, 'h': h, 't': replaced_t}

            return replaced_item
        elif view == 'ht':
            return self.replace_entity(self.replace_entity(item, 'h'), 't')
        else:
            raise Exception('View is not legal.')

    def next_train(self):
        if self.epoch == 0:
            raise Exception('Program End!')
        if self.train_index == self.epoch_split:
            self.construct_training_tasks()
            self.train_index = 0
        task = self.meta_train_tasks[self.train_index]
        self.train_index += 1
        return task

    def next_test(self):
        return self.meta_test_tasks