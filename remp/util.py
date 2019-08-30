import os

import pandas as pd
import numpy as np
from scipy.special import comb


def suffix(df, column_suffix):
    columns = df.columns
    return df.rename(columns=dict((column, column + column_suffix) for column in columns))


class FixedErrorOracle(object):
    def __init__(self, error_rate, repeats=5, threshold=4):
        hit = np.arange(threshold, repeats + 1)
        true_prob = (np.exp(np.log(error_rate) * hit + np.log(1 - error_rate) * (repeats - hit)) * comb(repeats, hit)).sum()
        error_rate = 1 - error_rate
        false_prob = (np.exp(np.log(error_rate) * hit + np.log(1 - error_rate) * (repeats - hit)) * comb(repeats, hit)).sum()
        self.false_threshold = false_prob
        self.true_threshold = 1 - true_prob

    def get_answer(self, true_answer):
        point = np.random.rand()
        if point >= self.true_threshold:
            return true_answer
        elif point <= self.false_threshold:
            return not true_answer
        else:
            return None


def get_column(df, column):
    return np.array([str(i) for i in df[column]], dtype=np.str)


def matching_quality(result_table, gold_table, s1='s1', s2='s2'):
    if len(result_table) > 0:
        tp = len(pd.merge(result_table, gold_table, on=[s1, s2], how='inner'))
        (p, r) = (tp / len(result_table), tp / len(gold_table))
        if p == 0.0 or r == 0.0:
            f = p = r = 0.0
        else:
            f = 2 * p * r / (p + r)
    else:
        f = p = r = 0.0
    return pd.DataFrame([[p, r, f]], index=['result'], columns=['precision', 'recall', 'f-measure'])


def inverse_index(l):
    return dict((l[idx], idx) for idx in range(0, len(l)))


class DeltaDict(object):
    def __init__(self, d):
        self.base = d
        self.data = {}

    def __getitem__(self, index):
        return self.data[index] if index in self.data else self.base[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def items(self):
        return self.data.items()

    def apply(self):
        for (k, v) in self.data.items():
            self.base[k] = v
        return self.base


def inverse_functionality(df, index, attr):
    return df[attr].nunique() / max(1, df[[index, attr]].groupby(by=attr)[index].count().sum())


def prepare_cache_folder(base_dir, task_name):
    import os
    import shutil
    target_folder = os.path.join(base_dir, task_name)
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)
    return target_folder


class CacheDecoreator(object):
    def __init__(self, base_dir, task_name):
        self.base_dir = base_dir
        self.task_name = task_name

    def category(self, cate):
        def real_func(func):
            def wrapper(*args, **kwargs):
                import os
                cache_name = os.path.join(self.base_dir, cate, self.task_name + '.pkl')
                if os.path.exists(cache_name):
                    return pd.read_pickle(cache_name)
                else:
                    import time
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    pd.to_pickle(result, cache_name)
                    pd.to_pickle({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time}, 
                        os.path.join(self.base_dir, cate, self.task_name + '.time.pkl'))
                return result
            return wrapper
        return real_func

    def read_cache(self, cate):
        return pd.read_pickle(os.path.join(self.base_dir, cate, self.task_name + '.pkl'))

    def read_time(self, cate):
        return pd.read_pickle(os.path.join(self.base_dir, cate, self.task_name + '.time.pkl'))
