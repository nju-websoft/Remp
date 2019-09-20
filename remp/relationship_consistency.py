import numpy as np
import pandas as pd
from scipy.special import comb
from .util import suffix


def relationship_consistency_mle(M, N, cnt, L=None):
    zeta_p = []
    for m, n in np.unique(np.array([M, N]), axis=1).T:
        l_size = min(m, n) + 1
        A = np.log(comb(m, np.arange(0, l_size)) * comb(
            n, np.arange(0, l_size)))
        i = np.arange(0, l_size * l_size) // l_size
        j = np.arange(0, l_size * l_size) % l_size
        P = (A[i] - A[j])[i > j] / (j - i)[i > j]
        zeta_p.append(P)
    zeta_p = np.exp(np.sort(np.unique(np.concatenate(zeta_p))))
    zeta_p = np.concatenate([np.array([0.0]), zeta_p, np.array([zeta_p.max() + 1])])
    zeta_p = (zeta_p[0:-1] + zeta_p[1:]) * 0.5

    best_l = np.zeros((len(L), len(zeta_p)), dtype=np.int)
    for i in range(0, len(L)):
        (m, n, l, c) = (M[i], N[i], L[i], cnt[i])
        ll = np.arange(0, l + 1)
        s = np.tile(np.log(comb(m, ll) * comb(n, ll)), len(zeta_p))
        s += np.tile(ll, len(zeta_p)) * np.repeat(zeta_p, len(ll))
        s = s.reshape((len(ll), len(zeta_p))).argmax(0)
        best_l[i] = s

    values = []
    scores = []
    for idx in range(0, len(zeta_p)):
        l = np.array(best_l[:, idx])
        if np.sum(l) == 0:
            continue

        B = np.sum(l * cnt)
        p1 = np.sum(np.log(comb(M, l)) * cnt)
        p2 = np.sum(np.log(comb(N, l)) * cnt)

        A1 = (M * cnt).sum()
        A2 = (N * cnt).sum()

        if B < A1 and B > 0:
            p3 = np.log(B / A1) * B + np.log(1 - B / A1) * (A1 - B)
        else:
            p3 = 0.0

        if B < A2 and B > 0:
            p4 = np.log(B / A2) * B + np.log(1 - B / A2) * (A2 - B)
        else:
            p4 = 0.0
        values.append((B / A1, B / A2))
        scores.append(p1 + p2 + p3 + p4)
    return tuple(values[np.argmax(scores)]) if len(scores) > 0 else (0.0, 0.0)


def relationship_consistency(M_in, M_p, r1, r2):
    r1_cnt = r1.groupby(by=['s', 'r'])['o'].count().reset_index()
    r2_cnt = r2.groupby(by=['s', 'r'])['o'].count().reset_index()
    inner = pd.merge(pd.merge(M_in, suffix(r1, '1')), suffix(r2, '2'))
    inner = pd.merge(inner, M_p.rename(columns={'s1': 'o1', 's2': 'o2'}))
    inner = inner.groupby(by=['s1', 's2', 'r1', 'r2'])[['o1', 'o2']].nunique(
        ).min(1).rename('l').reset_index()
    inner = pd.merge(pd.merge(inner, suffix(r1_cnt, '1')), suffix(r2_cnt, '2'))

    consistency = []
    for (r1, r2), df in inner.groupby(by=['r1', 'r2']):
        v1_extra = 0
        v2_extra = 0
        if (df['o1'] == df['o2']).all():
            (e1, e2) = (1.0, 1.0)
        else:
            lll = df[['o1', 'o2', 'l']].groupby(by=['o1', 'o2', 'l']).size()
            lll = lll.rename('count').reset_index()
            (e1, e2) = relationship_consistency_mle(
                lll['o1'], lll['o2'], lll['count'], lll['l'])
        consistency.append([r1, r2, e1, e2])

    consistency = pd.DataFrame(consistency, columns=['r1', 'r2', 'e1', 'e2'])

    return consistency
