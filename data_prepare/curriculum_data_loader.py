import os
from tqdm import tqdm
from data_prepare import DataLoader
from itertools import combinations


class CurriculumDataLoader(DataLoader):
    def load_extra(self):
        # this method is used to load extra files for data loader
        print('Load relation kl distance matrix information')
        self.kl_dist_h = self.read_json(os.path.join(self.dataset, 'kl_dist_h.json'))
        self.kl_dist_t = self.read_json(os.path.join(self.dataset, 'kl_dist_t.json'))
        self.kl_dist_ht = self.read_json(os.path.join(self.dataset, 'kl_dist_ht.json'))

    def construct_training_tasks(self):
        # construct training tasks with the data loader config
        # this method can be override to implement different strategies
        meta_train_epochs = []  # separate into 2 parts: train and test
        for epoch in tqdm(range(self.epoch)):
            # construct an epoch tasks
            epoch_tasks = []
            for task_id in range(self.task_num):
                # construct a task
                train_task = self.construct_task(self.train_relations, self.train_vec)
                epoch_tasks.append(train_task)
            # sort an epoch tasks by difficult
            sorted_epoch_tasks, epoch_difficulty = self.sort_tasks(epoch_tasks)
            meta_train_epochs.append({'tasks': sorted_epoch_tasks, 'difficulty': epoch_difficulty})

        # sort epochs
        sorted_meta_train_epochs = sorted(meta_train_epochs, key=lambda i: i['difficulty'])

        self.meta_train_tasks = [epoch['tasks'] for epoch in sorted_meta_train_epochs]

    def sort_tasks(self, epoch_tasks):
        for task in epoch_tasks:
            self.calc_difficulty(task)  # add keys of difficulty

        if self.difficulty_view == 'h':
            sorted_epoch_tasks = sorted(epoch_tasks, key=lambda i: i['kl_h'])
            epoch_difficulty = sum([task['kl_h'] for task in sorted_epoch_tasks])
        elif self.difficulty_view == 't':
            sorted_epoch_tasks = sorted(epoch_tasks, key=lambda i: i['kl_t'])
            epoch_difficulty = sum([task['kl_t'] for task in sorted_epoch_tasks])
        elif self.difficulty_view == 'ht':
            sorted_epoch_tasks = sorted(epoch_tasks, key=lambda i: i['kl_ht'])
            epoch_difficulty = sum([task['kl_ht'] for task in sorted_epoch_tasks])
        else:
            raise Exception('Difficulty view is not Correct!')

        return sorted_epoch_tasks, epoch_difficulty

    def calc_difficulty(self, task):
        task_relations = task['task_relations']
        task_pairs = list(combinations([task_relation['index'] for task_relation in task_relations], 2))
        task_pairs_kl_h = sum([self.kl_dist_h[item[0]][item[1]] for item in task_pairs])
        task_pairs_kl_t = sum([self.kl_dist_t[item[0]][item[1]] for item in task_pairs])
        task_pairs_kl_ht = sum([self.kl_dist_ht[item[0]][item[1]] for item in task_pairs])
        task['kl_h'] = task_pairs_kl_h
        task['kl_t'] = task_pairs_kl_t
        task['kl_ht'] = task_pairs_kl_ht




