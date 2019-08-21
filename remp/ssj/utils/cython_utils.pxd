
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from ssj.index.inverted_index_cy cimport InvertedIndexCy


ctypedef bool (*compfnptr)(double, double) nogil

cdef void tokenize_lists(lvalues, rvalues,
                         tokenizer,
                         vector[vector[int]]& ltokens,
                         vector[vector[int]]& rtokens)

cdef void build_inverted_index(vector[vector[int]]& token_vectors,
                               InvertedIndexCy inv_index)

cdef int get_comp_type(comp_op)
cdef compfnptr get_comparison_function(const int comp_type) nogil

cdef int int_min(int a, int b) nogil
cdef int int_max(int a, int b) nogil
