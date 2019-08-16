from cython.parallel import prange
from cpython.string cimport PyString_AsString

# import unidecode
from lxml import etree


def parse_xml(str string):
    try:
        return etree.fromstring('<x>' + string + '</x>').xpath('/x/text()')[0]
    except:
        return string
#
# def normalize(string):
#     return unidecode.unidecode(parse_xml(str(string).lower()))

cimport numpy as np
import numpy as np

cpdef np.ndarray[str] apply_integrate_f(object[:] values):
    cdef Py_ssize_t i, n = len(values)
    cdef char * res[n]
    cdef char * raw_str[n]
    for i in range(n):
        raw_str[i] = PyString_AsString(values[i])
    for i in prange(n, nogil=True):
        res[i] = raw_str[i]
    return res

cdef char * table[256*256]

for i in range(0, 255):
    try:
        y = __import__('unidecode.x%03x' % (i), [], [], ['data'])
        for j in range(0, len(y)):
            table[256 * i + j] = PyString_AsString(y)
    except ModuleNotFoundError:
        pass
#
# def unidecode(str):
#     def _mapchar(char):
#         codepoint = ord(char)