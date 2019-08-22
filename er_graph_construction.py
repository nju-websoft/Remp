import sys
from os import listdir, path, mkdir
import time

import pandas as pd
import numpy as np
import networkx as nx
import unidecode
from tqdm import tqdm_notebook
import swifter

import py_stringmatching as sm

from remp.util import prepare_cache_folder, matching_quality, CacheDecoreator, suffix
from remp.matching import exactly_string_matching_minimum
from remp.property_alignment import attribute_alignment
from remp.similarity_vector import construct_similarity_vector
from remp.topk_filtering import filter_one_way
from remp.ssj import jaccard_join

from datasets import PickleDataset


def main():
    if len(sys.argv) < 2:
        print('command: python er_graph_construction.py task_name cache_dir')
        return
    name = sys.argv[1]
    if len(sys.argv) >= 3:
        cache_base_dir = sys.argv[2]
    else:
        cache_base_dir = '/mnt/hdd/Cache'
    print('program start', name, cache_base_dir)

    # init cache
    cache = CacheDecoreator(cache_base_dir + '/', name)

    print('load dataset, time=%f' % time.time())
    dataset = PickleDataset(name, cache_base_dir + '/Dataset/' + name)

    print('label normalization, time=%f' % time.time())
    (l1, l2) = dataset.label
    labels_1 = dataset.attributes_1[dataset.attributes_1['a'] == l1][['s', 'v']]
    labels_2 = dataset.attributes_2[dataset.attributes_2['a'] == l2][['s', 'v']]

    labels_1['v'] = labels_1['v'].apply(str).apply(unidecode.unidecode).str.lower()
    labels_2['v'] = labels_2['v'].apply(str).apply(unidecode.unidecode).str.lower()

    print('exactly string matching, time=%f' % time.time())
    @cache.category('M_s')
    def do_exactly_string_matching_minimum(name):
        return exactly_string_matching_minimum(labels_1, labels_2)
    esmm = do_exactly_string_matching_minimum(name)

    print('attribute matching, time=%f' % time.time())
    if len(set(dataset.attributes_1['a']) ^ set(dataset.attributes_2['a'])) == 0:
        aa = None
    else:
        @cache.category('M_at')
        def do_attribute_matching():
            return attribute_alignment(
                dataset.attributes_1[dataset.attributes_1['a'] != l1], 
                dataset.attributes_2[dataset.attributes_2['a'] != l2], esmm)
        aa = do_attribute_matching()

    print('candidate matches, time=%f' % time.time())
    m_c_cahce_folder = cache_base_dir + '/M_c/' + name
    if path.exists(m_c_cahce_folder + '.time.pkl'):
        pair_files = [m_c_cahce_folder + '/' + f for f in listdir(m_c_cahce_folder)]
    else:
        start_time = time.time()
        if not path.exists(m_c_cahce_folder):
            mkdir(m_c_cahce_folder)
        tokenizer = sm.WhitespaceTokenizer(return_set=True)
        num_pairs, pair_files = jaccard_join(labels_1['v'], labels_2['v'], labels_1['s'], labels_2['s'], 
            tokenizer, 0.3, m_c_cahce_folder + '/', n_jobs=-1)
        end_time = time.time()
        pd.to_pickle({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time}, m_c_cahce_folder + '.time.pkl')

    if len(pair_files) < 100:
        @cache.category('M_p')
        def do_pair_pruning_small():
            # load candidate matches
            M_c = []
            for f in pair_files:
                M_c.append(pd.read_pickle(f))
            M_c = pd.DataFrame(sum(M_c, []), columns=['s1', 's2', 'score'])[['s1', 's2']].drop_duplicates()

            # construct similarity vector
            sv = construct_similarity_vector(dataset.attributes_1, dataset.attributes_2, M_c, aa)

            # compute M_p
            M_p_last = None
            disagr = 1.0
            s1 = set(dataset.attributes_1['s'])
            for k in range(1, 100):
                bounds, unresolved = filter_one_way(sv.fillna(0.0), k, 0)
                bounds, unresolved = filter_one_way(sv.loc[reversed(unresolved)].fillna(0.0), k, 0, bounds)
                bounds, unresolved = filter_one_way(sv.fillna(0.0), k, 1)
                bounds, unresolved = filter_one_way(sv.loc[reversed(unresolved)].fillna(0.0), k, 1, bounds)
                M_p = pd.DataFrame(list(unresolved), columns=['s1', 's2'])

                tc_size = 0
                for cc in nx.connected_components(nx.from_pandas_edgelist(M_p, 's1', 's2')):
                    left = cc & s1
                    right = cc - s1
                    if left == 1 or right == 1:
                        tc_size += left * right
                    else:
                        tc_size += len(pd.merge(pd.merge(M_c, pd.DataFrame({'s1': list(left)})), pd.DataFrame({'s2': list(right)})))
                new_disagr = 1 - len(M_p) / tc_size
                # if new_disagr > disagr:
                    # break
                disagr = new_disagr
                M_p_last = M_p
                print(k, disagr)
                print(matching_quality(M_p, dataset.link))
            return M_p_last
        M_p = do_pair_pruning_small()
if __name__ == "__main__":
    main()