import json
from tqdm import tqdm
from util.wikidata_query import WikiDataQuerier

train_file = './train.json'
test_file = './test.json'

with open(train_file, 'r') as f:
    train_items = json.load(f)

with open(test_file, 'r') as f:
    test_items = json.load(f)

querier = WikiDataQuerier()

entity2hypers = {}


def get_hyper(dataset):
    for relation_id, items in tqdm(dataset.items()):
        entity_list = set()
        for item in items:
            entity_list.add(item['h']['id'])
            entity_list.add(item['t']['id'])
        entity_list = list(entity_list)
        hyper_results = querier.fetch_hypernym_labels_by_entity_ids(entity_list)['results']['bindings']

        for res in hyper_results:
            entity_id = res['item']['value'].split('/')[-1]
            entity_label = res['itemLabel']['value']
            class_id = res['class']['value'].split('/')[-1]
            class_label = res['classLabel']['value']

            if entity_id not in entity2hypers:
                hyper_set = set()
                hyper_set.add(class_id + ' ' + class_label)
                entity2hypers[entity_id] = hyper_set
            else:
                hyper_set = entity2hypers[entity_id]
                hyper_set.add(class_id + ' ' + class_label)
                entity2hypers[entity_id] = hyper_set


print('fetch hyper of train set')
get_hyper(train_items)
print('fetch hyper of test set')
get_hyper(test_items)

e2h = {}
for entity_id, hyper_set in entity2hypers.items():
    hyper_list = []
    for hyper in list(hyper_set):
        class_id = hyper.split(' ')[0]
        class_label = hyper.split(' ')[1:]
        hyper_list.append({'id': class_id, 'label': class_label})
    e2h[entity_id] = hyper_list
with open('./e2h.json', 'w') as f:
    json.dump(e2h, f, indent=4)


