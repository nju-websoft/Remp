# set similarity join

import pandas as pd
import pyprind

from libc.math cimport ceil, floor, round, sqrt, trunc
from libcpp.vector cimport vector
from libcpp.set cimport set as oset
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as omap
from libcpp.pair cimport pair

from pickle import dump
from os import getpid

from ssj.index.position_index_cy cimport PositionIndexCy
from ssj.utils.cython_utils cimport compfnptr,\
    get_comparison_function, get_comp_type, int_max, int_min, tokenize_lists


cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()
    cdef int sum_of_size = size1 + size2
    if sum_of_size == 0:
        return 1.0
    if size1 == 0 or size2 == 0:
        return 0.0
    cdef int overlap = 0
    while i < size1 and j < size2:
        if tokens1[i] == tokens2[j]:
            overlap += 1
            i += 1
            j += 1
        elif tokens1[i] < tokens2[j]:
            i += 1
        else:
            j += 1
    return (overlap * 1.0) / <double>(sum_of_size - overlap)


def set_sim_join_cy(lvalues, rvalues, lkeys, rkeys,
                    tokenizer, sim_measure, threshold, comp_op,
                    allow_empty, show_progress, output_dir=''):
    cdef long long pair_cnt = 0
    cdef vector[string] pair_files

    cdef vector[vector[int]] ltokens, rtokens
    tokenize_lists(lvalues, rvalues, tokenizer, ltokens, rtokens)

    cdef int sim_type
    sim_type = get_sim_type(sim_measure)

    cdef PositionIndexCy index = PositionIndexCy()
    index = build_position_index(ltokens, sim_type, threshold, allow_empty)

    cdef omap[int, int] candidate_overlap, overlap_threshold_cache
    cdef vector[pair[int, int]] candidates
    cdef vector[int] tokens
    cdef pair[int, int] cand, entry
    cdef int k, j, m, i, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound
    cdef double sim_score, overlap_score
    cdef fnptr sim_fn
    cdef compfnptr comp_fn
    sim_fn = get_sim_function(sim_type)
    comp_fn = get_comparison_function(get_comp_type(comp_op))

    output_rows = []

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rvalues))
    pid = getpid()
    print('process id:', pid)

    for i in range(rtokens.size()):
        tokens = rtokens[i]
        m = tokens.size()

        if allow_empty and m == 0:
            for j in index.l_empty_ids:
                output_row = [lkeys[j], rkeys[i]]
                output_rows.append(output_row)
            continue

        prefix_length = get_prefix_length(m, sim_type, threshold)
        size_lower_bound = int_max(get_size_lower_bound(m, sim_type, threshold),
                                   index.min_len)
        size_upper_bound = int_min(get_size_upper_bound(m, sim_type, threshold),
                                   index.max_len)

        for size in range(size_lower_bound, size_upper_bound + 1):
            overlap_threshold_cache[size] = get_overlap_threshold(size, m, sim_type, threshold)

        for j in range(prefix_length):
            if index.index.find(tokens[j]) == index.index.end():
                continue
            candidates = index.index[tokens[j]]
            for cand in candidates:
                current_overlap = candidate_overlap[cand.first]
                if current_overlap != -1:
                    cand_num_tokens = index.size_vector[cand.first]

                    # only consider candidates satisfying the size filter
                    # condition.
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound:

                        if m - j <= cand_num_tokens - cand.second:
                            overlap_upper_bound = m - j
                        else:
                            overlap_upper_bound = cand_num_tokens - cand.second

                        # only consider candidates for which the overlap upper
                        # bound is at least the required overlap.
                        if (current_overlap + overlap_upper_bound >=
                                overlap_threshold_cache[cand_num_tokens]):
                            candidate_overlap[cand.first] = current_overlap + 1
                        else:
                            candidate_overlap[cand.first] = -1

        for entry in candidate_overlap:
            if entry.second > 0:
                sim_score = sim_fn(ltokens[entry.first], tokens)

                if comp_fn(sim_score, threshold):
                    output_row = [lkeys[entry.first], rkeys[i]]
                    output_rows.append(output_row)

        candidate_overlap.clear()
        overlap_threshold_cache.clear()

        if show_progress:
            prog_bar.update()
        if len(output_rows) > 1000000:
            with open((output_dir + '/%d-%d.pkl') % (pid, i), 'wb') as f:
                dump(output_rows, f)
            pair_cnt += len(output_rows)
            pair_files.push_back(((output_dir + '/%d-%d.pkl') % (pid, i)).encode('utf-8'))
            output_rows = []

    with open((output_dir + '%d.pkl') % pid, 'wb') as f:
        pair_cnt += len(output_rows)
        pair_files.push_back(((output_dir + '/%d.pkl') % (pid)).encode('utf-8'))
        dump(output_rows, f)

    return pair_cnt, pair_files

cdef PositionIndexCy build_position_index(vector[vector[int]]& token_vectors,
                               int& sim_type, double& threshold,
                               bool allow_empty):
    cdef PositionIndexCy pos_index = PositionIndexCy()
    cdef vector[int] tokens, size_vector
    cdef int prefix_length, token, i, j, m, n=token_vectors.size(), min_len=100000, max_len=0
    cdef omap[int, vector[pair[int, int]]] index
    cdef vector[int] empty_l_ids
    for i in range(n):
        tokens = token_vectors[i]
        m = tokens.size()
        size_vector.push_back(m)
        prefix_length = get_prefix_length(m, sim_type, threshold)
        for j in range(prefix_length):
            index[tokens[j]].push_back(pair[int, int](i, j))
        if m > max_len:
            max_len = m
        if m < min_len:
            min_len = m
        if allow_empty and m == 0:
            empty_l_ids.push_back(i)

    pos_index.set_fields(index, size_vector, empty_l_ids,
                         min_len, max_len, threshold)
    return pos_index


cdef int get_prefix_length(int& num_tokens, int& sim_type, double& threshold) nogil:
    return <int>(num_tokens - ceil(threshold * num_tokens) + 1.0)

cdef int get_size_lower_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    return <int>ceil(threshold * num_tokens)

cdef int get_size_upper_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    return <int>floor(num_tokens / threshold)

cdef int get_overlap_threshold(int& l_num_tokens, int& r_num_tokens, int& sim_type, double& threshold) nogil:
    return <int>ceil((threshold / (1 + threshold)) * (l_num_tokens + r_num_tokens))

ctypedef double (*fnptr)(const vector[int]&, const vector[int]&) nogil

cdef fnptr get_sim_function(int& sim_type) nogil:
    return jaccard

cdef int get_sim_type(sim_measure):
    return 2

