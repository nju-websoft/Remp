import numpy as np
import pandas as pd
from scipy.special import comb
from .util import suffix


def relationship_consistency_mle(v1, v2, cnt):
    import itertools
    l_max = np.array([v1, v2]).min(0)
    values = []
    scores = []
    for k in itertools.product(*[range(0, m + 1) for m in l_max]):
        l = np.array(k)
        B = np.sum(k * cnt)
        if B == 0 or B == np.sum(l_max * cnt):
            continue
        p1 = np.sum(np.log(comb(v1, k)) * cnt)
        p2 = np.sum(np.log(comb(v2, k)) * cnt)

        A1 = (v1 * cnt).sum()
        A2 = (v2 * cnt).sum()
        p3 = np.log(B / A1) * B + np.log(1 - B / A1) * (A1 - B)
        p4 = np.log(B / A2) * B + np.log(1 - B / A2) * (A2 - B)
        values.append((B / A1, B / A2))
        scores.append(p1 + p2 + p3 + p4)
    return tuple(values[np.argmax(scores)])


def relationship_consistency(M_in, r1, r2):
    shared_relations = pd.merge(M_in, suffix(r1, '1'))
    shared_relations = pd.merge(shared_relations,
                                M_in.rename(columns={'s1': 'o1', 's2': 'o2'}))
    shared_relations = pd.merge(shared_relations, suffix(r2, '2'))
    shared_relations = shared_relations[['r1', 'r2']].drop_duplicates()
    # TODO: remove entities which don't appear in M_p
    r1_cnt = r1.groupby(by=['s', 'r'])['o'].count().reset_index()
    r2_cnt = r2.groupby(by=['s', 'r'])['o'].count().reset_index()
    forward = pd.merge(M_in, suffix(r1_cnt, '1'), how='left')
    forward = pd.merge(forward, suffix(r2_cnt, '2'), how='left').fillna(0.0)
    forward = pd.merge(shared_relations, forward)

    consistency = []
    for (r1, r2), df in forward.groupby(by=['r1', 'r2']):
        if (df['o1'] == df['o2']).all():
            (e1, e2) = (1.0, 1.0)
        else:
            lll = df[['o1', 'o2']].groupby(by=['o1', 'o2']).agg('size')
            lll = lll.rename('count').reset_index()
            (e1, e2) = relationship_consistency_mle(
                lll['o1'], lll['o2'], lll['count'])
        consistency.append([r1, r2, e1, e2])
    consistency = pd.DataFrame(consistency, columns=['r1', 'r2', 'e1', 'e2'])

    return consistency
