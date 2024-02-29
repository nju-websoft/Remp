#include <iostream>
#include <functional>
#include <omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tokenizer/QgramTokenizer.hpp"
#include "tokenizer/WhitespaceTokenizer.hpp"
#include "similarity_measure/jaccard.hpp"
#include "similarity_measure/cosine.hpp"

namespace py = pybind11;

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
py::array_t<double> array_similarty_function(py::list X, py::list Y)
{
  py::array_t<double> result(std::min(X.size(), Y.size()));
  py::buffer_info result_buf = result.request();
  double *output = static_cast<double *>(result_buf.ptr);

  int N = std::min(X.size(), Y.size());

  PyObject **x_ptr = ((PyListObject *)X.ptr())->ob_item;
  PyObject **y_ptr = ((PyListObject *)Y.ptr())->ob_item;

  PyOjbectSimilarityFunction<Measure, Tokenizer> func;

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    output[i] = func(x_ptr[i], y_ptr[i]);
  }

  return result;
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


PYBIND11_MODULE(string_matching, m)
{
  m.def("array_whitespace_jaccard", &array_similarty_function<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>);
  m.def("array_qgram_jaccard_1", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer1>);
  m.def("array_qgram_jaccard_2", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer2>);
  m.def("array_qgram_jaccard_3", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer3>);
  m.def("array_qgram_jaccard_4", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer4>);
  m.def("array_qgram_jaccard_5", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer5>);
  m.def("array_qgram_jaccard_6", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer6>);
  m.def("array_qgram_jaccard_7", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer7>);
  m.def("array_qgram_jaccard_8", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer8>);
  m.def("array_qgram_jaccard_9", &array_similarty_function<similarity_measure::Jaccard, QgramTokenizer9>);

  m.def("array_whitespace_cosine", &array_similarty_function<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>);
  m.def("array_qgram_cosine_1", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer1>);
  m.def("array_qgram_cosine_2", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer2>);
  m.def("array_qgram_cosine_3", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer3>);
  m.def("array_qgram_cosine_4", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer4>);
  m.def("array_qgram_cosine_5", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer5>);
  m.def("array_qgram_cosine_6", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer6>);
  m.def("array_qgram_cosine_7", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer7>);
  m.def("array_qgram_cosine_8", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer8>);
  m.def("array_qgram_cosine_9", &array_similarty_function<similarity_measure::Cosine, QgramTokenizer9>);

  m.def("whitespace_jaccard", &pair_similarty_function<similarity_measure::Jaccard, tokenizer::WhitespaceTokenizer>);
  m.def("qgram_jaccard_1", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer1>);
  m.def("qgram_jaccard_2", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer2>);
  m.def("qgram_jaccard_3", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer3>);
  m.def("qgram_jaccard_4", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer4>);
  m.def("qgram_jaccard_5", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer5>);
  m.def("qgram_jaccard_6", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer6>);
  m.def("qgram_jaccard_7", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer7>);
  m.def("qgram_jaccard_8", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer8>);
  m.def("qgram_jaccard_9", &pair_similarty_function<similarity_measure::Jaccard, QgramTokenizer9>);

  m.def("whitespace_cosine", &pair_similarty_function<similarity_measure::Cosine, tokenizer::WhitespaceTokenizer>);
  m.def("qgram_cosine_1", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer1>);
  m.def("qgram_cosine_2", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer2>);
  m.def("qgram_cosine_3", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer3>);
  m.def("qgram_cosine_4", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer4>);
  m.def("qgram_cosine_5", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer5>);
  m.def("qgram_cosine_6", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer6>);
  m.def("qgram_cosine_7", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer7>);
  m.def("qgram_cosine_8", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer8>);
  m.def("qgram_cosine_9", &pair_similarty_function<similarity_measure::Cosine, QgramTokenizer9>);
}