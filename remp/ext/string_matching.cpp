#include <boost/python/numpy.hpp>
#include "tokenizer/QgramTokenizer.hpp"
#include "tokenizer/WhitespaceTokenizer.hpp"
#include "similarity_measure/jaccard.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

np::ndarray array_bigram_jaccard(np::ndarray X, np::ndarray Y);

#include "array_jaccard.ipp"

BOOST_PYTHON_MODULE(string_matching)
{
  Py_Initialize();
  np::initialize();
  p::def("array_bigram_jaccard", array_bigram_jaccard);
  p::def("array_whitespace_jaccard", array_jaccard< tokenizer::WhitespaceTokenizer<wchar_t> >);
  p::def("array_qgram_jaccard_2", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 2> >);
  p::def("array_qgram_jaccard_3", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 3> >);
  p::def("array_qgram_jaccard_4", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 4> >);
  p::def("array_qgram_jaccard_5", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 5> >);
  p::def("array_qgram_jaccard_6", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 6> >);
  p::def("array_qgram_jaccard_7", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 7> >);
  p::def("array_qgram_jaccard_8", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 8> >);
  p::def("array_qgram_jaccard_9", array_jaccard< tokenizer::QgramTokenizer<wchar_t, 9> >);
}
