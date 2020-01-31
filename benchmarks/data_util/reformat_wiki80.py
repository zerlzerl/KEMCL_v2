import json
import os

# input files
all_wiki_relations_file = '../wikidata/all_wiki_relations.json'
wiki80_rel2id_file = './original_data/wiki80_rel2id.json'
wiki80_train_file = './original_data/wiki80_train.txt'
wiki80_val_file = './original_data/wiki80_val.txt'
# output files
all_wiki80_relations_file = './all_relations.json'
train_relations_file = './train_relations.json'
test_relations_file = './test_relations.json'
train_file = './train.json'
test_file = './test.json'

# load all wiki relations
all_wiki_relations = {}  # dict
with open(all_wiki_relations_file, 'r') as f:
    all_wiki_relations = json.load(f)
print('All wikidata relations number: %d' % len(all_wiki_relations))

all_wiki80_relations = {}
with open(wiki80_rel2id_file, 'r') as f:
    all_wiki80_relations = json.load(f)
print('All wiki80 relations number: %d' % len(all_wiki80_relations))

all_wiki80_relations_list = []
for key in all_wiki80_relations:
    if key == 'original network':
        all_wiki80_relations_list.append('original broadcaster')
    else:
        all_wiki80_relations_list.append(key)

wiki_id_dict = {}
wiki_label_dict = {}
for relation_id in all_wiki_relations:
    wiki_id_dict[relation_id] = all_wiki_relations[relation_id]
    wiki_label_dict[all_wiki_relations[relation_id]['label']] = all_wiki_relations[relation_id]
print('Lenght of rid2rel dict: %d' % len(wiki_id_dict))
print('Lenght of label2rel dict: %d' % len(wiki_label_dict))

train_relation_num = 64
all_relations_detail = {}
train_relations_detail = {}
test_relations_detail = {}

for index in range(len(all_wiki80_relations_list)):
    rel = wiki_label_dict[all_wiki80_relations_list[index]]
    rel['index'] = index
    all_relations_detail[rel['id']] = rel
    if index < train_relation_num:
        train_relations_detail[rel['id']] = rel
    else:
        test_relations_detail[rel['id']] = rel

with open(all_wiki80_relations_file, 'w') as f:
    json.dump(all_relations_detail, f, indent=4)
with open(train_relations_file, 'w') as f:
    json.dump(train_relations_detail, f, indent=4)
with open(test_relations_file, 'w') as f:
    json.dump(test_relations_detail, f, indent=4)

# 读入wiki80 train
wiki80_train_items = []
wiki80_all_r2items = {}
with open(wiki80_train_file, 'r') as f:
    for line in f:
        item_dict = json.loads(line)
        wiki80_train_items.append(item_dict)
        relation_label = item_dict['relation']
        if relation_label == 'original network':
            relation_label = 'original broadcaster'
        relation_id = wiki_label_dict[relation_label]['id']
        if relation_id not in wiki80_all_r2items:
            wiki80_all_r2items[relation_id] = [item_dict]
        else:
            wiki80_all_r2items[relation_id].append(item_dict)

wiki80_val_items = []
with open(wiki80_val_file, 'r') as f:
    for line in f:
        item_dict = json.loads(line)
        wiki80_val_items.append(item_dict)
        relation_label = item_dict['relation']
        if relation_label == 'original network':
            relation_label = 'original broadcaster'
        relation_id = wiki_label_dict[relation_label]['id']
        if relation_id not in wiki80_all_r2items:
            wiki80_all_r2items[relation_id] = [item_dict]
        else:
            wiki80_all_r2items[relation_id].append(item_dict)

wiki80_train_r2items = {}
wiki80_test_r2items = {}

for relation_id in train_relations_detail:
    wiki80_train_r2items[relation_id] = wiki80_all_r2items[relation_id]

for relation_id in test_relations_detail:
    wiki80_test_r2items[relation_id] = wiki80_all_r2items[relation_id]

# SAVE
with open(train_file, 'w') as f:
    json.dump(wiki80_train_r2items, f, indent=4)
with open(test_file, 'w') as f:
    json.dump(wiki80_test_r2items, f, indent=4)

print('end.')


#
# wiki80_train_items = {}
# with open(wiki80_train_file, 'r') as f:
#     for line in f:
#         train_item = json.loads(line, encoding='utf-8')
#         token_list = train_item['token']
#         head_dict = train_item['h']
#         tail_dict = train_item['t']
#         relation_label = train_item['relation']
#         if relation_label in wiki_label2rel:
#             relation_id = wiki_label2rel[relation_label]['id']
#         elif relation_label is 'original broadcaster':
#             relation_id = wiki_label2rel[relation_label]['id']
#
#         if relation_id not in wiki80_train_items:
#             wiki80_train_items[relation_id] = [{'token': token_list, 'h': head_dict, 't': tail_dict,
#                                                 'r': {'label': train_item['relation'], 'id': relation_id}}]
#         else:
#             wiki80_train_items[relation_id].append({'token': token_list, 'h': head_dict, 't': tail_dict,
#                                                     'r': {'label': train_item['relation'], 'id': relation_id}})
