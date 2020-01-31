import json
import os

# input files
all_wiki_relations_file = '../wikidata/all_wiki_relations.json'
fewrel_train_rel2id_file = '../fewrel/fewrel_train_rel2id.json'
fewrel_val_rel2id_file = '../fewrel/fewrel_val_rel2id.json'
fewrel_train_file = '../fewrel/fewrel_train.txt'
fewrel_val_file = '../fewrel/fewrel_val.txt'
# output files
all_relations_file = '../fewrel/all_relations.json'
train_relations_file = '../fewrel/train_relations.json'
test_relations_file = '../fewrel/test_relations.json'
train_file = '../fewrel/train.json'
test_file = '../fewrel/test.json'

# load all wiki relations
all_wiki_relations = {}  # dict
with open(all_wiki_relations_file, 'r') as f:
    all_wiki_relations = json.load(f)
print('All wikidata relations number: %d' % len(all_wiki_relations))

fewrel_train_rel2id = {}
with open(fewrel_train_rel2id_file, 'r') as f:
    fewrel_train_rel2id = json.load(f)

fewrel_test_rel2id = {}
with open(fewrel_val_rel2id_file, 'r') as f:
    fewrel_test_rel2id = json.load(f)

fewrel_all_rel2id_detail = {}
fewrel_train_rel2id_detail = {}
for relation_id in fewrel_train_rel2id:
    index = fewrel_train_rel2id[relation_id]
    rel_detail = all_wiki_relations[relation_id]
    rel_detail['index'] = index
    fewrel_train_rel2id_detail[relation_id] = rel_detail
    fewrel_all_rel2id_detail[relation_id] = rel_detail

fewrel_test_rel2id_detail = {}
for relation_id in fewrel_test_rel2id:
    index = fewrel_test_rel2id[relation_id]
    rel_detail = all_wiki_relations[relation_id]
    rel_detail['index'] = index + 64
    fewrel_test_rel2id_detail[relation_id] = rel_detail
    fewrel_all_rel2id_detail[relation_id] = rel_detail

fewrel_train_items = {}
with open(fewrel_train_file, 'r') as f:
    for line in f:
        item_dict = json.loads(line)
        train_items = {
            'token': item_dict['token'],
            'h': item_dict['h'],
            't': item_dict['t']
        }
        item_rel_id = item_dict['relation']
        if item_rel_id not in fewrel_train_items:
            fewrel_train_items[item_rel_id] = [train_items]
        else:
            fewrel_train_items[item_rel_id].append(train_items)

fewrel_test_items = {}
with open(fewrel_val_file, 'r') as f:
    for line in f:
        item_dict = json.loads(line)
        test_items = {
            'token': item_dict['token'],
            'h': item_dict['h'],
            't': item_dict['t']
        }
        item_rel_id = item_dict['relation']
        if item_rel_id not in fewrel_test_items:
            fewrel_test_items[item_rel_id] = [test_items]
        else:
            fewrel_test_items[item_rel_id].append(test_items)

# save
with open(all_relations_file, 'w') as f:
    json.dump(fewrel_all_rel2id_detail, f, indent=4)

with open(train_relations_file, 'w') as f:
    json.dump(fewrel_train_rel2id_detail, f, indent=4)

with open(test_relations_file, 'w') as f:
    json.dump(fewrel_test_rel2id_detail, f, indent=4)

with open(train_file, 'w') as f:
    json.dump(fewrel_train_items, f, indent=4)

with open(test_file, 'w') as f:
    json.dump(fewrel_test_items, f, indent=4)
print()