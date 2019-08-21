__version__ = '0.3.0'

# determine whether to use available cython implementations
__use_cython__ = True

# import join methods
from ssj.join.jaccard_join_cy import jaccard_join_cy as jaccard_join