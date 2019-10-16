import numpy as np
import pandas as pd
import itertools

from .relationship_consistency import relationship_consistency
from .util import suffix


def connected_components_id(e1, e2):
    cluster = np.arange(e1.shape[0])
    for i in range(1, e1.shape[0]):
        cluster1 = cluster[(e1 == e1[i]).argmax()]
        cluster2 = cluster[(e2 == e2[i]).argmax()]
        cluster[e1 == e1[i]] = min(cluster1, cluster2)
        cluster[e2 == e2[i]] = min(cluster1, cluster2)
        cluster[cluster == cluster1] = min(cluster1, cluster2)
        cluster[cluster == cluster2] = min(cluster1, cluster2)
    return cluster


def matching_probability(o1, o2, p, u1, u2, e1, e2, show_score=False):
    pt = np.zeros((len(u1), len(u2)))
    for i in range(p.shape[0]):
        pt[np.argmax(u1 == o1[i]), np.argmax(u2 == o2[i])] = p[i]

    scores = []
    for tu in itertools.permutations(range(0, len(u2) + 1), len(u1)):
        prob = 1.0
        l = 0
        for i in range(len(u1)):
            if tu[i] < len(u2):
                prob *= pt[i, tu[i]] / (1 - pt[i, tu[i]])
                l += 1
        prob *= (e1 / (1 - e1)) ** l
        prob *= (e2 / (1 - e2)) ** l
        scores.append([tu, prob])
    if show_score:
        print(len(u1), len(u2))
        print(list(itertools.permutations(range(0, len(u2) + 1), len(u1))))
        print(pt)
        print(scores)
    Z = sum(k[1] for k in scores)
    for i1 in range(0, len(u1)):
        for i2 in range(0, len(u2)):
            margin = sum(k[1] for k in scores if k[0][i1] == i2) / Z
            p[(o1 == u1[i1]) & (o2 == u2[i2])] = margin
    return p


def propagate2neighbor(edges, source, target, consistency):
    ff_dict = consistency.set_index(['r1', 'r2']).to_dict()
    forward = pd.Series(0.0, index=edges.index)
    e1_dict = ff_dict['e1']
    e2_dict = ff_dict['e2']
    for ((s1, s2, r1, r2), df) in edges.groupby(by=source):
        if (r1, r2) not in e1_dict:
            continue
        e1 = e1_dict[(r1, r2)]
        e2 = e2_dict[(r1, r2)]
        soften = 0.000001
        e1 = (soften + e1) / (1.0 + soften + soften)
        e2 = (soften + e2) / (1.0 + soften + soften)
        index = df.index
        values = df[target].values
        o1 = values[:, 0]
        o2 = values[:, 1]
        p = (soften + values[:, 2]) / (1.0 + soften + soften)
        cluster = connected_components_id(o1, o2)
        for i in range(0, len(cluster)):
            if cluster[i] == i:
                co1 = o1[cluster == i]
                co2 = o2[cluster == i]
                po1 = p[cluster == i]

                u1 = np.unique(co1)
                u2 = np.unique(co2)
                if len(u1) ** len(u2) > 1000000:
                    print(len(u1), len(u2))
                    continue
                if len(u1) < len(u2):
                    forward.loc[index[cluster == i]] = matching_probability(
                        co1, co2, po1, u1, u2, e1, e2)
                else:
                    forward.loc[index[cluster == i]] = matching_probability(
                        co2, co1, po1, u2, u1, e2, e1)
    return forward


def construct_er_graph(r1, r2, M_in, M_p, lsv):
    def backward(r):
        return r.rename(columns=dict(s='t', o='s')).rename(columns=dict(t='o'))
    b1 = backward(r1)
    b2 = backward(r2)
    # mle
    ff = relationship_consistency(M_in, M_p, r1, r2)
    fb = relationship_consistency(M_in, M_p, r1, b2)
    bf = relationship_consistency(M_in, M_p, b1, r2)
    bb = relationship_consistency(M_in, M_p, b1, b2)

    rr = pd.merge(ff[['r1', 'r2']], bb[['r1', 'r2']], how='outer')
    rr_edges = pd.merge(pd.merge(suffix(r1, '1'), rr),
                        lsv.rename(columns={'p': 'sp'}))
    rr_edges = pd.merge(rr_edges, suffix(r2, '2'))
    rr_edges = pd.merge(rr_edges, lsv.rename(
        columns={'s1': 'o1', 's2': 'o2', 'p': 'op'}))

    rb = pd.merge(fb[['r1', 'r2']], bf[['r1', 'r2']], how='outer')
    rb_edges = pd.merge(pd.merge(suffix(r1, '1'), rr),
                        lsv.rename(columns={'p': 'sp'}))
    rb_edges = pd.merge(rb_edges, suffix(b2, '2'))
    rb_edges = pd.merge(rb_edges, lsv.rename(
        columns={'s1': 'o1', 's2': 'o2', 'p': 'op'}))

    rr_edges['forward'] = propagate2neighbor(
        rr_edges, ['s1', 's2', 'r1', 'r2'], ['o1', 'o2', 'op'], ff)
    rr_edges['backward'] = propagate2neighbor(
        rr_edges, ['o1', 'o2', 'r1', 'r2'], ['s1', 's2', 'sp'], bb)
    rb_edges['forward'] = propagate2neighbor(
        rb_edges, ['s1', 's2', 'r1', 'r2'], ['o1', 'o2', 'op'], fb)
    rb_edges['backward'] = propagate2neighbor(
        rb_edges, ['o1', 'o2', 'r1', 'r2'], ['s1', 's2', 'sp'], bf)

    cols = ['s1', 's2', 'o1', 'o2', 'forward', 'backward']
    p_erg = pd.concat([rr_edges, rb_edges])[cols].groupby(
        by=['s1', 's2', 'o1', 'o2']).agg(
            {'forward': 'max', 'backward': 'max'}).reset_index()
    return p_erg
