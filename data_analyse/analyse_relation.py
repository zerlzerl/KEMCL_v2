from util.general_utils import *
from collections import OrderedDict
from util.wikidata_query import WikiDataQuerier
from prettytable import PrettyTable

relation = 'P931'
file_name = relation + '_th.pkl'
top_k = 20
triple_hypernyms = read_pickle(file_name)
relation_json = read_json('../benchmarks/fewrel/all_relations.json')
querier = WikiDataQuerier()

print('Read %d triples with hypernyms from relation %s - %s' % (len(triple_hypernyms), relation, relation_json[relation]['label']))

triple_set = set()
h_entity_set = set()
t_entity_set = set()
h_hyper_set = set()
t_hyper_set = set()
h_entity_count = {}
t_entity_count = {}
h_hyper2entity = {}
t_hyper2entity = {}
h_hyper_count = {}
t_hyper_count = {}

for one_triple_hyper in triple_hypernyms:
    h = one_triple_hyper[0]
    t = one_triple_hyper[1]
    r = one_triple_hyper[2]
    h_hyper = one_triple_hyper[3]
    t_hyper = one_triple_hyper[4]

    triple_set.add((h, t, r))
    h_entity_set.add(h)
    t_entity_set.add(t)
    h_hyper_set.add(h_hyper)
    t_hyper_set.add(t_hyper)

    if h not in h_entity_count:
        h_entity_count[h] = 1
    else:
        h_entity_count[h] += 1

    if t not in t_entity_count:
        t_entity_count[t] = 1
    else:
        t_entity_count[t] += 1

    if h_hyper not in h_hyper2entity:
        hyper_entity_set = set()
        hyper_entity_set.add(h)
        h_hyper2entity[h_hyper] = hyper_entity_set

        h_hyper_count[h_hyper] = 1
    else:
        h_hyper2entity[h_hyper].add(h)
        h_hyper_count[h_hyper] += 1

    if t_hyper not in t_hyper2entity:
        hyper_entity_set = set()
        hyper_entity_set.add(t)
        t_hyper2entity[t_hyper] = hyper_entity_set

        t_hyper_count[t_hyper] = 1
    else:
        t_hyper2entity[t_hyper].add(t)
        t_hyper_count[t_hyper] += 1

# sort
h_entity_count = OrderedDict(sorted(h_entity_count.items(), key=lambda item: item[1], reverse=True))
t_entity_count = OrderedDict(sorted(t_entity_count.items(), key=lambda item: item[1], reverse=True))

h_hyper_count = OrderedDict(sorted(h_hyper_count.items(), key=lambda item: item[1], reverse=True))
t_hyper_count = OrderedDict(sorted(t_hyper_count.items(), key=lambda item: item[1], reverse=True))

h_hyper2entity = OrderedDict(sorted(h_hyper2entity.items(), key=lambda item: len(item[1]), reverse=True))
t_hyper2entity = OrderedDict(sorted(t_hyper2entity.items(), key=lambda item: len(item[1]), reverse=True))

print('Analysis of relation %s - %s:' % (relation, relation_json[relation]['label']))
print('\tTriples: %d' % len(triple_set))
print('\tHead entities: %d' % len(h_entity_set))
print('\tTail entities: %d' % len(t_entity_set))
print('\tHead-Hypernyms: %d' % len(h_hyper_set))
print('\tTail-Hypernyms: %d' % len(t_hyper_set))

top_k_h_hyper = []
top_k_t_hyper = []

index = 0
for k, v in h_hyper_count.items():
    top_k_h_hyper.append(k)
    index += 1
    if index == top_k:
        break

top_k_h_hyper_details = querier.fetch_labels_by_entity_ids(top_k_h_hyper)['results']['bindings']

index = 0
for k, v in t_hyper_count.items():
    top_k_t_hyper.append(k)
    index += 1
    if index == top_k:
        break

top_k_t_hyper_details = querier.fetch_labels_by_entity_ids(top_k_t_hyper)['results']['bindings']

print('\nThe Top %d head hypernyms:' % top_k)
top_head_table = PrettyTable(['wiki_id', 'wiki_name', 'occurrence num', 'proportion', 'hyponym num'])
total_proportion = 0.0
for top_h_hyper in top_k_h_hyper_details:
    wiki_id = top_h_hyper['item']['value'].split('/')[-1]
    wiki_name = top_h_hyper['itemLabel']['value']
    occurrence = h_hyper_count[wiki_id]
    proportion = occurrence / len(triple_hypernyms)
    hyponym_num = len(h_hyper2entity[wiki_id])
    top_head_table.add_row([wiki_id, wiki_name, occurrence, '%.2f%%' % (proportion * 100), hyponym_num])
    total_proportion += proportion
print(top_head_table)
print('The Top %d head hypernyms accounted for %.2f%% of the total' % (top_k, total_proportion * 100))

print('\nThe Top %d tail hypernyms:' % top_k)
top_tail_table = PrettyTable(['wiki_id', 'wiki_name', 'occurrence num', 'proportion', 'hyponym num'])
total_proportion = 0.0
for top_t_hyper in top_k_t_hyper_details:
    wiki_id = top_t_hyper['item']['value'].split('/')[-1]
    wiki_name = top_t_hyper['itemLabel']['value']
    occurrence = t_hyper_count[wiki_id]
    proportion = occurrence / len(triple_hypernyms)
    hyponym_num = len(t_hyper2entity[wiki_id])
    top_tail_table.add_row([wiki_id, wiki_name, occurrence, '%.2f%%' % (proportion * 100), hyponym_num])
    total_proportion += proportion
print(top_tail_table)
print('The Top %d tail hypernyms accounted for %.2f%% of the total' % (top_k, total_proportion * 100))







