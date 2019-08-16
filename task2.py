from remp.util import prepare_cache_folder
import pandas as pd
import numpy as np

import datasets
dataset = datasets.DBLP_ACM()
dataset.attributes_1['a'] = dataset.attributes_1['a'].astype('category')
dataset.attributes_2['a'] = dataset.attributes_2['a'].astype('category')
dataset.relations_1['r'] = dataset.relations_1['r'].astype('category')
dataset.relations_2['r'] = dataset.relations_2['r'].astype('category')
link = dataset.link
labels_1 = dataset.attributes_1[dataset.attributes_1['a'] == 'label'][['s', 'v']]
labels_2 = dataset.attributes_2[dataset.attributes_2['a'] == 'label'][['s', 'v']]

import unidecode
labels_1['v'] = labels_1['v'].apply(unidecode.unidecode).str.lower()
labels_2['v'] = labels_2['v'].apply(unidecode.unidecode).str.lower()


import ssj
import py_stringmatching as sm
tokenizer = sm.WhitespaceTokenizer(return_set=True)
cache_folder = prepare_cache_folder('/mnt/hdd/Cache/', 'DA') + '/'
last_part = ssj.jaccard_join(labels_1, labels_2, 's', 's', 'v', 'v', tokenizer, 0.3, cache_folder, n_jobs=4)

from os import listdir
pair_parts = ['/mnt/hdd/Cache/DA/' + fname for fname in listdir('/mnt/hdd/Cache/DA/')]

pairs = []
import pickle
for pair_part in pair_parts:
    with open(pair_part, 'rb') as f:
        pairs += pickle.load(f)


import pandas as pd
candidate_part = pd.DataFrame(pairs, columns=['s1', 's2', 'score'])[['s1', 's2']]

from remp import util
util.matching_quality(candidate_part, link)

from remp.similarity_vector import construct_similarity_vector, similarity_func_default, construct_similarity_vector_from_tuples

sv = construct_similarity_vector(dataset.attributes_1, dataset.attributes_2, candidate_part)