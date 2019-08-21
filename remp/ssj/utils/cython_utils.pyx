
import pandas as pd

from ssj.utils.token_ordering import gen_token_ordering_for_tables,\
    order_using_token_ordering

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map as omap

from ssj.index.inverted_index_cy cimport InvertedIndexCy


cdef void tokenize_lists(lvalues, rvalues, tokenizer,
                         vector[vector[int]]& ltokens,
                         vector[vector[int]]& rtokens):

    token_ordering = gen_token_ordering_for_tables(
                         lvalues, rvalues, tokenizer)

    for lstr in lvalues:
        py_tokens = order_using_token_ordering(
                        tokenizer.tokenize(lstr), token_ordering)
        ltokens.push_back(py_tokens)

    for rstr in rvalues:
        py_tokens = order_using_token_ordering(
                        tokenizer.tokenize(rstr), token_ordering)
        rtokens.push_back(py_tokens)


cdef void build_inverted_index(vector[vector[int]]& token_vectors,
                               InvertedIndexCy inv_index):
    cdef vector[int] tokens, size_vector
    cdef int i, j, m, n=token_vectors.size()
    cdef omap[int, vector[int]] index
    for i in xrange(n):
        tokens = token_vectors[i]
        m = tokens.size()
        size_vector.push_back(m)
        for j in range(m):
            index[tokens[j]].push_back(i)
    inv_index.set_fields(index, size_vector)


cdef int get_comp_type(comp_op):
    if comp_op == '<':
        return 0
    elif comp_op == '<=':
        return 1
    elif comp_op == '>':
        return 2
    elif comp_op == '>=':
        return 3
    elif comp_op == '=':
        return 4


cdef compfnptr get_comparison_function(const int comp_type) nogil:
    if comp_type == 0:
        return lt_compare
    elif comp_type == 1:
        return le_compare
    elif comp_type == 2:
        return gt_compare
    elif comp_type == 3:
        return ge_compare
    elif comp_type == 4:
        return eq_compare


cdef bool eq_compare(double val1, double val2) nogil:
    return val1 == val2


cdef bool le_compare(double val1, double val2) nogil:
    return val1 <= val2


cdef bool lt_compare(double val1, double val2) nogil:
    return val1 < val2


cdef bool ge_compare(double val1, double val2) nogil:
    return val1 >= val2


cdef bool gt_compare(double val1, double val2) nogil:
    return val1 > val2


cdef int int_min(int a, int b) nogil:
    return a if a <= b else b


cdef int int_max(int a, int b) nogil:
    return a if a >= b else b
