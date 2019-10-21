import pandas as pd
import unidecode
import py_stringmatching as sm
from .similarity_vector import construct_similarity_vector, construct_similarity_list
from .topk_filtering import filter_one_way
from .ssj import jaccard_join


def exactly_string_matching(labels_1, labels_2,
                            s1='s', s2='s', l1='v', l2='v'):
    labels_1 = labels_1[[s1, l1]].rename(columns={s1: 's1', l1: 'v'})
    labels_2 = labels_2[[s2, l2]].rename(columns={s2: 's2', l2: 'v'})
    groups = pd.merge(labels_1, labels_2)
    return groups[['s1', 's2']]


def exactly_string_matching_minimum(labels_1, labels_2,
                                    s1='s', s2='s', l1='v', l2='v'):
    labels_1 = labels_1[[s1, l1]].rename(columns={s1: 's1', l1: 'v'})
    labels_2 = labels_2[[s2, l2]].rename(columns={s2: 's2', l2: 'v'})
    groups = pd.merge(labels_1, labels_2)
    groups = groups.set_index('v')
    groups_cnt = groups.groupby(by='v')['s1'].count()
    exactly_matches = groups.loc[groups_cnt[groups_cnt == 1].index]
    exactly_matches = exactly_matches.reset_index()
    exactly_matches = exactly_matches[['s1', 's2']].drop_duplicates()
    return exactly_matches


def initial_matching(dataset):
    (l1, l2) = dataset.label
    labels_1 = dataset.attributes_1[
        dataset.attributes_1['a'] == l1][['s', 'v']]
    labels_2 = dataset.attributes_2[
        dataset.attributes_2['a'] == l2][['s', 'v']]

    labels_1['v'] = labels_1['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    labels_2['v'] = labels_2['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    return exactly_string_matching_minimum(labels_1, labels_2)


def candidate_matching(dataset):
    import tempfile
    cache_base_dir = tempfile.mkdtemp('remp')
    (l1, l2) = dataset.label
    labels_1 = dataset.attributes_1[
        dataset.attributes_1['a'] == l1][['s', 'v']]
    labels_2 = dataset.attributes_2[
        dataset.attributes_2['a'] == l2][['s', 'v']]

    labels_1['v'] = labels_1['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    labels_2['v'] = labels_2['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    tokenizer = sm.WhitespaceTokenizer(return_set=True)
    num_pairs, pair_files = jaccard_join(labels_1['v'], labels_2['v'], 
        labels_1['s'], labels_2['s'], tokenizer, 0.3, cache_base_dir + '/', n_jobs=-1)
    M_c = [pd.read_pickle(f) for f in pair_files]
    return pd.DataFrame(sum(M_c, []), columns=['s1', 's2']).drop_duplicates()


def pruned_matching(dataset, M_c, M_at, k):
    sv = construct_similarity_vector(dataset.attributes_1, dataset.attributes_2, M_c, None)
    bounds, unresolved = filter_one_way(sv.fillna(0.0), k, 0)
    bounds, unresolved = filter_one_way(sv.loc[reversed(unresolved)].fillna(0.0), k, 0, bounds)
    bounds, unresolved = filter_one_way(sv.fillna(0.0), k, 1)
    bounds, unresolved = filter_one_way(sv.loc[reversed(unresolved)].fillna(0.0), k, 1, bounds)
    M_p = pd.DataFrame(list(unresolved), columns=['s1', 's2'])
    return M_p


def prior_probabilities(dataset, M_pruned):
    import tempfile
    cache_base_dir = tempfile.mkdtemp('remp')
    (l1, l2) = dataset.label
    labels_1 = dataset.attributes_1[
        dataset.attributes_1['a'] == l1][['s', 'a', 'v']]
    labels_2 = dataset.attributes_2[
        dataset.attributes_2['a'] == l2][['s', 'a', 'v']]

    labels_1['v'] = labels_1['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    labels_2['v'] = labels_2['v'].apply(
        str).apply(unidecode.unidecode).str.lower()
    return construct_similarity_list(labels_1, labels_2, M_pruned).rename(
        columns={'sim': 'p'}).groupby(by=['s1', 's2'])['p'].mean().reset_index()