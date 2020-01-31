from config import Config
from model import BaseModel
from data_prepare import DataLoader
con = Config()
con.set_device_type('cuda:0')  # 'cuda:x' or 'cpu'
con.set_dataset('../benchmarks/fewrel/')
con.set_model(BaseModel)
con.set_model_config(input_dim=1024, output_dim=5)
con.set_data_loader(DataLoader)
con.set_random_seed(317)
con.set_epoch(1000)
con.set_test_epoch(20)
con.set_n_way(5)
con.set_k_spt(1)
con.set_k_qry(15)
con.set_task_num(32)
con.set_sentence_encode_mode('full')
con.set_meta_lr(1e-2)
con.set_update_lr(0.4)
con.set_update_step(5)
con.set_update_step_test(10)
con.set_test_step(50)
con.set_opt_method('Adam')

con.set_meta_export_dir('../checkpoint/')
# init dependencies
con.init()

# train model
con.run()
