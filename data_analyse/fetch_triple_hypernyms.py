# this .py script is used to analyse the support entities of hypernyms
import os

from util.wikidata_query import WikiDataQuerier
from util.general_utils import *
from tqdm import tqdm

querier = WikiDataQuerier()

relations = read_json('./relations.json')

print('load %d relations' % len(relations))

data_file_list = os.listdir('./')


for relation in tqdm(relations):
    if relation + '_th.pkl' in data_file_list:
        print('relation %s have already fetched' % relation)
        continue
    else:
        triples_hypernyms = querier.fetch_triples_hypernyms_by_relation(relation)['results']['bindings']  # list
        print('fetch %d results for relation %s' % (len(triples_hypernyms), relation))
        triples_hypernyms_list = []
        for one_res in triples_hypernyms:
            h = one_res['s']['value'].split('/')[-1]
            t = one_res['o']['value'].split('/')[-1]
            r = relation
            h_hyper = one_res['sclass']['value'].split('/')[-1]
            t_hyper = one_res['oclass']['value'].split('/')[-1]

            # statistics
            triples_hypernyms_list.append([h, t, r, h_hyper, t_hyper])

        dump_pickle(relation + '_th.pkl', triples_hypernyms_list)





