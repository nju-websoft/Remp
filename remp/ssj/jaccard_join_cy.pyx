# jaccard join
from joblib import delayed, Parallel
import pandas as pd

from remp.ssj.generic_helper import get_num_processes_to_launch, split_table

from remp.ssj.set_sim_join_cy import set_sim_join_cy


def jaccard_join_cy(lvalues, rvalues, lkeys, rkeys,
                    tokenizer, threshold, output_dir='./', comp_op='>=',
                    allow_empty=True, n_jobs=1, show_progress=True):

    # set return_set flag of tokenizer to be True, in case it is set to False
    revert_tokenizer_return_set_flag = False
    if not tokenizer.get_return_set():
        tokenizer.set_return_set(True)
        revert_tokenizer_return_set_flag = True

    lvalues = lvalues.values
    rvalues = rvalues.values
    lkeys = lkeys.values
    rkeys = rkeys.values

    # computes the actual number of jobs to launch.
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(rvalues))

    if n_jobs <= 1:
        # if n_jobs is 1, do not use any parallel code.
        pair_cnt = set_sim_join_cy(lvalues, rvalues,
                                       lkeys, rkeys,
                                       tokenizer, 'JACCARD',
                                       threshold, comp_op, allow_empty,
                                       show_progress, output_dir)
    else:
        # if n_jobs is above 1, split the right table into n_jobs splits and
        # join each right table split with the whole of left table in a separate
        # process.
        rvalue_splits = split_table(rvalues, n_jobs)
        rkey_splits = split_table(rkeys, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(set_sim_join_cy)(
                                          lvalues, rvalue_splits[job_index],
                                          lkeys, rkey_splits[job_index],
                                          tokenizer, 'JACCARD',
                                          threshold, comp_op, allow_empty,
                                      (show_progress and (job_index==n_jobs-1)), output_dir)
                                          for job_index in range(n_jobs))
        pair_cnt = sum(cnt for cnt, _ in results), sum(pairs for _, pairs in results)

    pair_cnt = pair_cnt[0], [pairs.decode('utf-8') for pairs in pair_cnt[1]]
    # revert the return_set flag of tokenizer, in case it was modified.
    if revert_tokenizer_return_set_flag:
        tokenizer.set_return_set(False)

    return pair_cnt
