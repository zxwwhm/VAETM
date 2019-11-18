import numpy as np
import pickle
from typing import List

ENTITY_CATEGORY = 'WikiData.KB.100d'
DIM = 100

class WikiLabel(object):
    """docstring for WikiLabel"""
    def __init__(self, id:int, label:str, title:str):
        self.id = id
        self.label = label
        self.title = title

    def __repr__(self):
        return 'Wikidata[id={}, label={}, title={}]'.format(self.id, self.label, self.title)

from mysql.connector import (connection)
cnx = connection.MySQLConnection(user='root', password='root',
                                 host='10.2.2.60',
                                 database='wikidata')


def list_wiki_labels(start_id:int, size:int)->List[WikiLabel]:
    results = []
    query = ("SELECT id, label, title_en_lower FROM label_title "
             "WHERE id BETWEEN %s AND %s AND INSTR(`title_en`, ' ') = 0")
    end_id = start_id + size - 1
    print('loading wiki_labels[{}, {}]...'.format(start_id, end_id))
    cursor = cnx.cursor()
    cursor.execute(query, (start_id, end_id))
    for mid, label, title_en_lower in cursor:
        if title_en_lower.isalnum():
            wl = WikiLabel(mid, label, title_en_lower)
            results.append(wl)
    cursor.close()
    if len(results)>0:
        print('\t last:', str(results[-1]))
    return results



def load_kb_mat(vec_file_name:str) -> np.memmap:
    print("Loading k2b vectors from", vec_file_name, '...')
    kb_mat = np.memmap(vec_file_name , dtype='float32', mode='r')
    print("size of kb2vec_file:", kb_mat.shape)
    num = kb_mat.shape[0]
    kb_mat.resize(int(num/DIM), DIM)
    print(type(kb_mat))
    print("size of kb_mat:", kb_mat.shape)
    print('mat[0] =\n', kb_mat[0,:])
    return kb_mat

def load_entity_labels()->List:
    entities = []
    # total num: 2098,2733
    for i in range(2099):
        start_id = i*10000
        sub_list = list_wiki_labels(start_id, 10000)
        entities.extend(sub_list)
    print('# of entities selected:', len(entities))
    return entities

def main():
    kb2vec_file = "/home/lcw2/share/embeddings/{}/entity2vec.bin".format(ENTITY_CATEGORY)
    mat = load_kb_mat(kb2vec_file)
    entities = load_entity_labels()
    cnx.close()
    
    print('assembling entity_embedding dict...')
    entity_embedding = dict()
    for entity in entities:
        title = entity.title
        if title in entity_embedding:
            continue
        mid = entity.id
        entity_embedding[title] = mat[mid]

    print('pickling entity_embedding...')
    with open('data/kb2vec/{}.pickle'.format(ENTITY_CATEGORY), 'wb') as f:
        pickle.dump(entity_embedding, f, pickle.HIGHEST_PROTOCOL)
    print('finished!')



if __name__ == '__main__':
    main()