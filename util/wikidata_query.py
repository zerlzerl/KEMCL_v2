import requests
from util.user_agent_tool import get_headers
import json

class WikiDataQuerier:
    def __init__(self):
        self.wikidata_query_endpoint = 'https://query.wikidata.org/sparql'

    def get_query_result(self, query_str):
        params = {'format': 'json', 'query': query_str}
        headers = get_headers()
        query_result = requests.post(self.wikidata_query_endpoint, data=params, headers=headers)
        if query_result.status_code == 200:
            return json.loads(query_result.content)
        else:
            print('error code: %d' % query_result.status_code)
            return None

    def fetch_all_wikidata_relations(self):
        query_str = 'SELECT ?property ?propertyType ?propertyLabel ?propertyDescription ?propertyAltLabel WHERE {' \
                    '?property wikibase:propertyType ?propertyType.' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }' \
                    '}' \
                    'ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), \'P\')))'
        return self.get_query_result(query_str)

    def fetch_triples_hypernyms_by_relation(self, relation_wiki_id):
        query_str = 'SELECT ?s ?p ?o ?sclass ?oclass ' \
                    'WHERE {' \
                    '?s wdt:%s ?o. ' \
                    '?s wdt:P31 ?sclass. ' \
                    '?o wdt:P31 ?oclass.' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }' \
                    '} LIMIT 1000000' % relation_wiki_id
        # print(query_str)
        return self.get_query_result(query_str)

    def fetch_hypernym_by_entity(self, entity_wiki_id):
        query_str = 'SELECT ?hypernym ' \
                    'WHERE {' \
                    'wd:%s wdt:P31 ?hypernym. ' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }' \
                    '}' % entity_wiki_id
        return self.get_query_result(query_str)

    def fetch_hypernym_labels_by_entity_ids(self, ids):
        """
        ids is a list of ids of wikidata entities
        :param ids:
        :return:
        """
        query_str = 'SELECT ?item ?itemLabel ?class ?classLabel WHERE {' \
                    'VALUES ?item { %s }' \
                    '?item wdt:P31 ?class;' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }' \
                    '}' % ' '.join(['wd:%s' % wiki_id for wiki_id in ids])
        # print(query_str)
        return self.get_query_result(query_str)

    def fetch_labels_by_entity_ids(self, ids):
        query_str = 'SELECT ?item ?itemLabel WHERE {' \
                    'VALUES ?item { %s }' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }' \
                    '}' % ' '.join(['wd:%s' % wiki_id for wiki_id in ids])
        return self.get_query_result(query_str)

    def fetch_instanceof_or_subclassof_relation_between_a_list_ids(self, ids):
        query_str = 'SELECT ?head ?tail WHERE {' \
                    'VALUES ?head { ' \
                    '%s' \
                    '}' \
                    '?head wdt:P31|wdt:P279 ?tail.' \
                    'FILTER (?tail IN ( %s ))' \
                    'SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }' \
                    '}' % (' '.join(['wd:%s' % wiki_id for wiki_id in ids]), ','.join(['wd:%s' % wiki_id for wiki_id in ids]))
        return self.get_query_result(query_str)


if __name__ == '__main__':
    # fetch all relations information from wikidata and store in local file system
    querier = WikiDataQuerier()
    all_relations = querier.fetch_all_wikidata_relations()['results']['bindings']
    print('fetch %d relations from wikidata.' % len(all_relations))
    rels = {}
    for relation in all_relations:
        rel = {}
        if 'property' in relation:
            property_id = relation['property']['value'].split('/')[-1]
            rel['id'] = property_id
        if 'propertyType' in relation:
            property_type = relation['propertyType']['value']
        if 'propertyLabel' in relation:
            property_label = relation['propertyLabel']['value']
            rel['label'] = property_label
        if 'propertyDescription' in relation:
            property_desc = relation['propertyDescription']['value']
            rel['description'] = property_desc
        if 'propertyAltLabel' in relation:
            property_alt_label = relation['propertyAltLabel']['value']
            rel['alt_label'] = property_alt_label
        rels[property_id] = rel

    with open('../wikidata/all_wiki_relations.json', 'w') as f:
        json.dump(rels, f, indent=4)

