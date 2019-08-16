import py_stringmatching as sm
import numpy as np

from remp import string_matching

words = ['kitten', 'sitting', 'sitting kitten']
n = len(words)
objarr = np.array(words, dtype=np.object)
print(string_matching.array_whitespace_jaccard(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print()
print(string_matching.array_qgram_jaccard_2(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_qgram_jaccard_3(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_qgram_jaccard_4(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_qgram_jaccard_5(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_qgram_jaccard_6(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_qgram_jaccard_7(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))
# print(string_matching.array_whitespace_jaccard(np.repeat(objarr, n), np.tile(objarr, n)).reshape((n, n)))