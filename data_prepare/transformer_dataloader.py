from data_prepare import DataLoader
import torch
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm


class TransformerDataLoader(DataLoader):
    def generate_embeddings(self, rel2items_dict, view=None):
        rel2vec_dict = {}
        for relation_id, train_items in tqdm(rel2items_dict.items()):
            emb_vec = torch.zeros(len(train_items), 3, 1024)
            symbol_pos = []
            for index, item in enumerate(train_items):
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
                    vec = torch.cat((sentence_vec.view(-1, 1024), first_ent_vec.view(-1, 1024), second_ent_vec.view(-1, 1024)), dim=0)
                    emb_vec[index] = vec
            rel2vec_dict[relation_id] = emb_vec.cpu().detach().numpy()
        return rel2vec_dict

