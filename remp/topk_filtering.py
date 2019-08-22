import numpy as np


def filter_group(vectors, unresolved, k):
    front = []
    bound = np.zeros(vectors.shape[0], dtype=np.bool)
    while len(unresolved) > 0:
        pick = unresolved[0]
        npred = ((vectors > vectors[pick]).any(1) & (vectors >= vectors[pick]).all(1)).sum()
        if npred >= k:
            bound[pick] = (vectors[pick] > 0.0).sum() > 1
            unresolved = unresolved[(vectors[unresolved] > vectors[pick]).any(1)]
        else:
            front.append(pick)
            unresolved = unresolved[1:]
    return front, vectors[bound]


def filter_one_way(vector_frame, k, way, bounds=None):
    if bounds is None:
        bounds = []
    reserveds = []

    for s1, group in vector_frame.groupby(by=vector_frame.index.names[way]):
        unresolved = np.arange(0, group.shape[0])
        single_valued = unresolved[(group.values[unresolved] > 0).sum(1) == 1]
        unresolved = unresolved[(group.values[unresolved] > 0).sum(1) > 1]

        for bound in bounds:
            for row in bound:
                unresolved = unresolved[(group.values[unresolved] > row).any(1)]
        unresolved = np.concatenate([single_valued, unresolved])
        if len(unresolved) > k:
            (reserved, new_bound) = filter_group(group.values, unresolved, k)
            bounds.append(new_bound)
            reserveds.append(group.index[reserved].values)
        else:
            reserveds.append(group.index[unresolved].values)
    return bounds, np.concatenate(reserveds)


def filter_two_way(vector_frame, k):
    front = []
    while len(vector_frame) > 0:
        v = vector_frame.head(1).values[0]
        idx = (vector_frame.values >= v).all(1)
        ancestors = vector_frame[idx]
        if ancestors.groupby(by='s1').size().max() > k or ancestors.groupby(by='s2').size().max() > k:
            vector_frame = vector_frame[~(vector_frame.values <= v).all(1)]
        else:
            front.append(ancestors.index.values)
            vector_frame = vector_frame[~idx]
    return np.concatenate(front)


