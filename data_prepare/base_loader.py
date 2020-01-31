class BaseLoader:
    def __init__(self, config):
        self.dataset = config['dataset']
        self.epoch = config['epoch']
        self.n_way = config['n_way']
        self.k_spt = config['k_spt']
        self.k_qry = config['k_qry']
        self.task_num = config['task_num']
        self.bert_service_ip = config['bert_service_ip']
        self.test_epoch = config['test_epoch']
        self.difficulty_view = config['difficulty_view']
        self.sentence_encode_mode = config['sentence_encode_mode']
        self.device = config['device']
        self.epoch_split = config['epoch_split']
        self.invalid_chars = ['\xa0', '\n', ' ', '\u3000', '\u2005']
        self.train_index = 0
        self.test_index = 0
        self.init()

    def init(self):
        # init data loader with config
        pass

    def next_train(self):
        # get data of task_num tasks(a whole epoch)
        pass

    def next_test(self):
        # get data of  a task
        pass

    def remove_invalid_token(self, token_list):
        for invalid_char in self.invalid_chars:
            token_list = [char for char in token_list if invalid_char not in char]
        return token_list
