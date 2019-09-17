#include <boost/python/numpy.hpp>
#include "tokenizer/QgramTokenizer.hpp"
#include "tokenizer/WhitespaceTokenizer.hpp"
#include "similarity_measure/jaccard.hpp"
#include "similarity_measure/cosine.hpp"

#include <iostream>
#include <functional>
#include <omp.h>

namespace p = boost::python;
namespace np = boost::python::numpy;

template<template<class> class Measure, template<typename> class Tokenizer>
struct PyOjbectSimilarityFunction {
  inline float operator() (PyObject * x, PyObject * y) {
    #ifdef _WIN32
    typename Measure<typename Tokenizer<int>::token_type>::container_type x_tokens, y_tokens;
    tokenizer_4(PyUnicode_4BYTE_DATA(x), x_tokens);
    tokenizer_4(PyUnicode_4BYTE_DATA(y), y_tokens);
    return measure_4.get_sim_score(x_tokens, y_tokens);
    #else
    int char_width = std::max(PyUnicode_KIND(x), PyUnicode_KIND(y));

    if (char_width == 1) {
      typename Measure<typename Tokenizer<char>::token_type>::container_type x_tokens, y_tokens;
      tokenizer_1(PyUnicode_1BYTE_DATA(x), x_tokens);
      tokenizer_1(PyUnicode_1BYTE_DATA(y), y_tokens);
      return measure_1.get_sim_score(x_tokens, y_tokens);
    } else if (char_width == 2) {
      typename Measure<typename Tokenizer<wchar_t>::token_type>::container_type x_tokens, y_tokens;
      tokenizer_2(PyUnicode_2BYTE_DATA(x), x_tokens);
      tokenizer_2(PyUnicode_2BYTE_DATA(y), y_tokens);
      return measure_2.get_sim_score(x_tokens, y_tokens);
    } else if (char_width == 4) {
      typename Measure<typename Tokenizer<int>::token_type>::container_type x_tokens, y_tokens;
      tokenizer_4(PyUnicode_4BYTE_DATA(x), x_tokens);
      tokenizer_4(PyUnicode_4BYTE_DATA(y), y_tokens);
      return measure_4.get_sim_score(x_tokens, y_tokens);
    } else {
      throw std::runtime_error("unexpected string format");
    }
    #endif
  }
private:
  Tokenizer<char> tokenizer_1;
  Tokenizer<wchar_t> tokenizer_2;
  Tokenizer<int> tokenizer_4;
  Measure<typename Tokenizer<char>::token_type> measure_1;
  Measure<typename Tokenizer<wchar_t>::token_type> measure_2;
  Measure<typename Tokenizer<int>::token_type> measure_4;
};


template<template<class> class Measure, template<typename> class Tokenizer>
np::ndarray array_similarty_function(np::ndarray X, np::ndarray Y) {
  if (X.get_nd() != 1 || Y.get_nd() != 1) {
    std::cout << "X and Y should be 1-dimentional array" << std::endl;
    throw std::runtime_error("X and Y should be 1-dimentional array");
  }

  if (X.shape(0) != Y.shape(0)) {
    std::cout << "X and Y should be same size" << std::endl;
    throw std::runtime_error("X and Y should be same size");
  }

  unsigned long N = X.shape(0);
  float *sims = new float[N];

  PyObject ** _X = reinterpret_cast<PyObject **>(X.get_data());
  PyObject ** _Y = reinterpret_cast<PyObject **>(Y.get_data());

  PyOjbectSimilarityFunction<Measure, Tokenizer> func;

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    sims[i] = func(_X[i], _Y[i]);
  }

  return np::from_data(sims, np::dtype::get_builtin<float>(),
    p::make_tuple(N), p::make_tuple(sizeof(float)), p::object());
}


template<template<class> class Measure, template<typename> class Tokenizer>
float pair_similarty_function(PyObject * x, PyObject * y) {
  PyOjbectSimilarityFunction<Measure, Tokenizer> func;

  return func(x, y);
}

template<class T> using QgramTokenizer1 = tokenizer::QgramTokenizer<1, T>;
template<class T> using QgramTokenizer2 = tokenizer::QgramTokenizer<2, T>;
template<class T> using QgramTokenizer3 = tokenizer::QgramTokenizer<3, T>;
template<class T> using QgramTokenizer4 = tokenizer::QgramTokenizer<4, T>;
template<class T> using QgramTokenizer5 = tokenizer::QgramTokenizer<5, T>;
template<class T> using QgramTokenizer6 = tokenizer::QgramTokenizer<6, T>;
template<class T> using QgramTokenizer7 = tokenizer::QgramTokenizer<7, T>;
template<class T> using QgramTokenizer8 = tokenizer::QgramTokenizer<8, T>;
template<class T> using QgramTokenizer9 = tokenizer::QgramTokenizer<9, T>;


BOOST_PYTHON_MODULE(string_matching)
{
  Py_Initialize();
  np::initialize();

  p::def("array_whitespace_jaccard", array_similarty_function<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>);
  p::def("array_qgram_jaccard_1", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer1>);
  p::def("array_qgram_jaccard_2", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer2>);
  p::def("array_qgram_jaccard_3", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer3>);
  p::def("array_qgram_jaccard_4", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer4>);
  p::def("array_qgram_jaccard_5", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer5>);
  p::def("array_qgram_jaccard_6", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer6>);
  p::def("array_qgram_jaccard_7", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer7>);
  p::def("array_qgram_jaccard_8", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer8>);
  p::def("array_qgram_jaccard_9", array_similarty_function<similarity_measure::Jaccard, QgramTokenizer9>);

  p::def("array_whitespace_cosine", array_similarty_function<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>);
  p::def("array_qgram_cosine_1", array_similarty_function<similarity_measure::Cosine, QgramTokenizer1>);
  p::def("array_qgram_cosine_2", array_similarty_function<similarity_measure::Cosine, QgramTokenizer2>);
  p::def("array_qgram_cosine_3", array_similarty_function<similarity_measure::Cosine, QgramTokenizer3>);
  p::def("array_qgram_cosine_4", array_similarty_function<similarity_measure::Cosine, QgramTokenizer4>);
  p::def("array_qgram_cosine_5", array_similarty_function<similarity_measure::Cosine, QgramTokenizer5>);
  p::def("array_qgram_cosine_6", array_similarty_function<similarity_measure::Cosine, QgramTokenizer6>);
  p::def("array_qgram_cosine_7", array_similarty_function<similarity_measure::Cosine, QgramTokenizer7>);
  p::def("array_qgram_cosine_8", array_similarty_function<similarity_measure::Cosine, QgramTokenizer8>);
  p::def("array_qgram_cosine_9", array_similarty_function<similarity_measure::Cosine, QgramTokenizer9>);

  p::def("whitespace_jaccard", pair_similarty_function<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>);
  p::def("qgram_jaccard_1", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer1>);
  p::def("qgram_jaccard_2", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer2>);
  p::def("qgram_jaccard_3", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer3>);
  p::def("qgram_jaccard_4", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer4>);
  p::def("qgram_jaccard_5", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer5>);
  p::def("qgram_jaccard_6", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer6>);
  p::def("qgram_jaccard_7", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer7>);
  p::def("qgram_jaccard_8", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer8>);
  p::def("qgram_jaccard_9", pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer9>);

  p::def("whitespace_cosine", pair_similarty_function<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>);
  p::def("qgram_cosine_1", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer1>);
  p::def("qgram_cosine_2", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer2>);
  p::def("qgram_cosine_3", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer3>);
  p::def("qgram_cosine_4", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer4>);
  p::def("qgram_cosine_5", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer5>);
  p::def("qgram_cosine_6", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer6>);
  p::def("qgram_cosine_7", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer7>);
  p::def("qgram_cosine_8", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer8>);
  p::def("qgram_cosine_9", pair_similarty_function<similarity_measure::Cosine, QgramTokenizer9>);
}