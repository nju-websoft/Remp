import multiprocessing
import operator

from six.moves import xrange


COMP_OP_MAP = {'>=': operator.ge,
               '>': operator.gt,
               '<=': operator.le,
               '<': operator.lt,
               '=': operator.eq,
               '!=': operator.ne}


def split_table(table, num_splits):
    splits = []
    split_size = 1.0/num_splits*len(table)
    for i in xrange(num_splits):
        splits.append(table[int(round(i*split_size)):
                            int(round((i+1)*split_size))])
    return splits


def get_num_processes_to_launch(n_jobs):
    # determine number of processes to launch parallely
    num_procs = n_jobs
    if n_jobs < 0:
        num_cpus = multiprocessing.cpu_count()
        num_procs = num_cpus + 1 + n_jobs
    return max(num_procs, 1)