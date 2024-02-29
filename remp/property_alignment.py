import py_stringmatching as sm
import pandas as pd
import numpy as np
import networkx as nx
from remp.string_matching import array_qgram_jaccard_2
from remp.util import suffix
import unidecode

bigram = sm.QgramTokenizer(qval=2, return_set=True)
jaccard = sm.Jaccard()


def similarity_func_default(tu):
    return jaccard.get_sim_score(
        bigram.tokenize(tu.v1), bigram.tokenize(tu.v2))


def attribute_alignment(attributes_1,
                        attributes_2, prior_alignment, similarity_func=None):
    (s, a, v) = ('s', 'a', 'v')
    # normalization
    attributes_1 = pd.merge(prior_alignment.rename(
        columns={s + '1': s})[['s']], attributes_1)
    attributes_2 = pd.merge(prior_alignment.rename(
        columns={s + '2': s})[['s']], attributes_2)
    attributes_1[v] = attributes_1[v].apply(str).apply(
        unidecode.unidecode).apply(str.lower)
    attributes_2[v] = attributes_2[v].apply(str).apply(
        unidecode.unidecode).apply(str.lower)

    # attribute value count
    a1_cnt = attributes_1[[a, s]].groupby(by=a)[s].count().reset_index()
    a1_cnt = suffix(a1_cnt, '1')
    a2_cnt = attributes_2[[a, s]].groupby(by=a)[s].count().reset_index()
    a2_cnt = suffix(a2_cnt, '2')

    # value bigram jaccard
    paired = pd.merge(prior_alignment, suffix(attributes_1, '1'))
    paired = pd.merge(paired, suffix(attributes_2, '2'))
    jaccard = array_qgram_jaccard_2(list(paired.v1.values), list(paired.v2.values))
    overlap_size = paired.assign(score=jaccard).groupby(
        by=[a + '1', a + '2'])['score'].agg(['sum', 'size']).reset_index()

    # attribute jaccard
    overlap_size = overlap_size[overlap_size['sum'] > 0]
    attr_jaccard = pd.merge(
        pd.merge(overlap_size, a1_cnt, how='left'),
        a2_cnt, how='left').fillna(0.0)
    attr_jaccard['weight'] = attr_jaccard['sum'] / \
        (attr_jaccard['s1'] + attr_jaccard['s2'] - attr_jaccard['size'])

    g = nx.from_pandas_edgelist(
        attr_jaccard[attr_jaccard['weight'] >= 0.05],
        source='a1', target='a2', edge_attr=['weight'])

    # maximum weight matching
    mwm = nx.max_weight_matching(g)

    # normalize the mwm
    mwm = list(mwm)
    a1 = list(a1_cnt['a1'].values)
    a2 = list(a2_cnt['a2'].values)
    for i in range(0, len(mwm)):
        if not (mwm[i][0] in a1 and mwm[i][1] in a2):
            mwm[i] = (mwm[i][1], mwm[i][0])

    return pd.DataFrame(mwm, columns=[a + '1', a + '2'])


def attribute_alignment_slow(attributes_1,
                             attributes_2,
                             prior_alignment,
                             similarity_func=similarity_func_default):
    (s, a, v) = ('s', 'a', 'v')
    # normalization
    attributes_1[v] = attributes_1[v].apply(str)
    attributes_2[v] = attributes_2[v].apply(str)

    # split the data frame by attributes
    groups1 = list(attributes_1.groupby(by=a))
    groups2 = list(attributes_2.groupby(by=a))

    # construct similarity graph
    g = nx.Graph()
    for a1, A1 in groups1:
        for a2, A2 in groups2:
            paired = prior_alignment
            paired = pd.merge(paired, A1[['s', 'v']].rename(
                columns={'s': 's1', 'v': 'v1'}), how='left')
            paired = pd.merge(paired, A2[['s', 'v']].rename(
                columns={'s': 's2', 'v': 'v2'}), how='left')
            total_cnt = np.sum(paired['v1'].notna() | paired['v2'].notna())
            paired = paired.dropna()
            if len(paired) > 0:
                score = paired.apply(
                    similarity_func_default, axis=1).sum() / total_cnt
                if score >= 0.05:
                    g.add_edge(a1, a2, weight=score)

    # maximum weight matching
    mwm = nx.max_weight_matching(g)

    # normalize the mwm
    mwm = list(mwm)
    a1 = [a for a, _ in groups1]
    a2 = [a for a, _ in groups2]
    for i in range(0, len(mwm)):
        if not (mwm[i][0] in a1 and mwm[i][1] in a2):
            mwm[i] = (mwm[i][1], mwm[i][0])

    return pd.DataFrame(mwm, columns=[s + '1', s + '2'])


def relation_jaccard(relations_1, relations_2, prior_matching):
    r1_cnt = relations_1[['r', 's']].groupby(by='r')['s'].count().reset_index()
    r1_cnt = suffix(r1_cnt, '1')
    r2_cnt = relations_2[['r', 's']].groupby(by='r')['s'].count().reset_index()
    r2_cnt = suffix(r2_cnt, '2')

    inner = pd.merge(prior_matching, suffix(relations_1, '1'))
    inner = inner[['s2', 'r1', 'o1']]
    sm2om = prior_matching.rename(columns={'s1': 'o1', 's2': 'o2'})
    inner = pd.merge(sm2om, inner)[['s2', 'r1', 'o2']]
    inner = pd.merge(inner, suffix(relations_2, '2'))
    s = inner[['r1', 'r2']].groupby(by=['r1', 'r2']).apply(len)
    s.name = 'overlap'

    s = pd.merge(s.reset_index(), r1_cnt)
    s = pd.merge(s.reset_index(), r2_cnt)
    s['jaccard'] = s['overlap'] / (s['s1'] + s['s2'] - s['overlap'])
    s = s[['r1', 'r2', 'jaccard']]
    return s


def attribute_jaccard(a1, a2, prior_matching):
    r1_cnt = a1[['a', 's']].groupby(by='a')['s'].count().reset_index()
    r1_cnt = suffix(r1_cnt, '1')
    r2_cnt = a2[['a', 's']].groupby(by='a')['s'].count().reset_index()
    r2_cnt = suffix(r2_cnt, '2')

    inner = pd.merge(prior_matching, suffix(a1, '1'))[['s2', 'a1', 'v1']]
    inner = inner.rename(columns={'v1': 'v2'})
    inner = pd.merge(inner, suffix(a2, '2'))
    s = inner[['a1', 'a2']].groupby(by=['a1', 'a2']).apply(len)
    s.name = 'overlap'

    s = pd.merge(s.reset_index(), r1_cnt)
    s = pd.merge(s.reset_index(), r2_cnt)
    s['jaccard'] = s['overlap'] / (s['s1'] + s['s2'] - s['overlap'])
    s = s[['a1', 'a2', 'jaccard']]
    return s
