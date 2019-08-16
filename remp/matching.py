import pandas as pd


def exactly_string_matching(labels_1, labels_2, s1='s', s2='s', l1='v', l2='v'):
    labels_1 = labels_1[[s1, l1]].rename(columns={s1: 's1', l1: 'v'})
    labels_2 = labels_2[[s2, l2]].rename(columns={s2: 's2', l2: 'v'})
    groups = pd.merge(labels_1, labels_2)
    return groups[['s1', 's2']]


def exactly_string_matching_minimum(labels_1, labels_2, s1='s', s2='s', l1='v', l2='v'):
    labels_1 = labels_1[[s1, l1]].rename(columns={s1: 's1', l1: 'v'})
    labels_2 = labels_2[[s2, l2]].rename(columns={s2: 's2', l2: 'v'})
    groups = pd.merge(labels_1, labels_2)
    groups = groups.set_index('v')
    groups_cnt = groups.groupby(by='v')['s1'].count()
    exactly_matches = groups.loc[groups_cnt[groups_cnt == 1].index].reset_index()[['s1', 's2']].drop_duplicates()
    return exactly_matches


