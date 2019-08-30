import numpy as np
import pandas as pd


def partial_order_maximal(vectors):
    unresolved = np.arange(0, len(vectors))
    is_maximal = np.zeros(len(vectors), dtype=np.bool)
    while len(unresolved) > 0:
        pick = unresolved[0]
        unresolved = unresolved[(vectors[unresolved] > vectors[pick]).any(1)]
        if not (vectors[unresolved] >= vectors[pick]).all(1).any():
            is_maximal[pick] = True
    return is_maximal


def filter_by_bounds(vector_frame, bounds):
    unresolved = np.arange(len(vector_frame))
    single_valued = unresolved[(vector_frame.values > 0).sum(1) <= 1]
    unresolved = unresolved[(vector_frame.values > 0).sum(1) > 1]
    for bound in bounds.values:
        unresolved = unresolved[
                (vector_frame.values[unresolved] > bound).any(1)]
    unresolved = np.sort(np.concatenate([single_valued, unresolved]))
    return pd.DataFrame(vector_frame.values[unresolved],
                        index=vector_frame.index[unresolved],
                        columns=vector_frame.columns)


def filter_group(vectors, unresolved, k):
    front = []
    bound = np.zeros(vectors.shape[0], dtype=np.bool)
    while len(unresolved) > 0:
        pick = unresolved[0]
        npred = ((vectors > vectors[pick]).any(1) &
                 (vectors >= vectors[pick]).all(1)).sum()
        if npred >= k:
            bound[pick] = (vectors[pick] > 0.0).sum() > 1
            unresolved = unresolved[(vectors[unresolved] >
                                     vectors[pick]).any(1)]
        else:
            front.append(pick)
            unresolved = unresolved[1:]
    return front, vectors[bound]


def filter_topk_one_way_df(vector_frame, k, way, show_progress=False):
    from tqdm import tqdm_notebook
    reserved_values = []
    reserved_indexs = []
    with tqdm_notebook(total=len(vector_frame.index),
                       disable=(not show_progress)) as pbar:
        group_idx_name = vector_frame.index.names[way]
        for s1, group in vector_frame.groupby(by=group_idx_name):
            unresolved = np.arange(0, group.shape[0])
            if len(unresolved) > k:
                (reserved, _) = filter_group(group.values, unresolved, k)
                reserved_values.append(group.values[reserved])
                reserved_indexs.append(group.index[reserved])
            else:
                reserved_values.append(group.values[unresolved])
                reserved_indexs.append(group.index[unresolved])
            pbar.update(group.shape[0])
    reserved_indexs = pd.MultiIndex.from_tuples(
                          np.concatenate(reserved_indexs), names=('s1', 's2'))
    return pd.DataFrame(
        np.concatenate(reserved_values),
        index=reserved_indexs,
        columns=vector_frame.columns)


def filter_one_way_df(vector_frame, k, way, bounds=None, show_progress=False):
    if bounds is None:
        bounds = []
    reserveds = []

    from tqdm import tqdm_notebook
    with tqdm_notebook(total=len(vector_frame.index)) as pbar:
        group_idx_name = vector_frame.index.names[way]
        for s1, group in vector_frame.groupby(by=group_idx_name):
            unresolved = np.arange(0, group.shape[0])
            is_single_valued = (group.values[unresolved] > 0).sum(1)
            single_valued = unresolved[is_single_valued == 1]
            unresolved = unresolved[is_single_valued > 1]

            for bound in bounds:
                for row in bound:
                    unresolved = unresolved[
                        (group.values[unresolved] > row).any(1)]
            unresolved = np.concatenate([single_valued, unresolved])
            if len(unresolved) > k:
                (reserved, new_bound) = filter_group(group.values,
                                                     unresolved, k)
                if len(new_bound) > 0:
                    bounds.append(new_bound)
                reserveds.append(pd.DataFrame(
                    group.values[reserved],
                    index=group.index[reserved],
                    columns=vector_frame.columns))
            else:
                reserveds.append(pd.DataFrame(
                    group.values[unresolved],
                    index=group.index[unresolved],
                    columns=vector_frame.columns))
            pbar.update(group.shape[0])
    bounds = np.concatenate(bounds)
    bounds = pd.DataFrame(bounds, columns=vector_frame.columns)
    return bounds, pd.concat(reserveds)


# def filter_one_way(vector_frame, k, way, bounds=None):
#     if bounds is None:
#         bounds = []
#     reserveds = []

#     for s1, group in vector_frame.groupby(by=vector_frame.index.names[way]):
#         unresolved = np.arange(0, group.shape[0])
#         single_valued = unresolved[(group.values[unresolved] > 0).sum(1)
#                                    == 1]
#         unresolved = unresolved[(group.values[unresolved] > 0).sum(1) > 1]

#         for bound in bounds:
#             for row in bound:
#                 unresolved = unresolved[(group.values[unresolved]
#                                          > row).any(1)]
#         unresolved = np.concatenate([single_valued, unresolved])
#         if len(unresolved) > k:
#             (reserved, new_bound) = filter_group(group.values, unresolved, k)
#             if len(new_bound) > 0:
#                 bounds.append(new_bound)
#             reserveds.append(group.index[reserved].values)
#         else:
#             reserveds.append(group.index[unresolved].values)
#     return bounds, np.concatenate(reserveds)


# def filter_two_way(vector_frame, k):
#     front = []
#     while len(vector_frame) > 0:
#         v = vector_frame.head(1).values[0]
#         idx = (vector_frame.values >= v).all(1)
#         ancestors = vector_frame[idx]
#         if ancestors.groupby(by='s1').size().max() > k or ancestors.groupby(
#               by='s2').size().max() > k:
#             vector_frame = vector_frame[~(vector_frame.values <= v).all(1)]
#         else:
#             front.append(ancestors.index.values)
#             vector_frame = vector_frame[~idx]
#     return np.concatenate(front)
