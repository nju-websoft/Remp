#include <boost/python/numpy.hpp>
#include <iostream>
#include <stdexcept>
#include <set>
#include <array>
#include <string>
#include <fstream>
#include <cinttypes>
#include <valarray>

#include <omp.h>

#include "similarity_measure/jaccard.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

np::ndarray array_bigram_jaccard(np::ndarray X, np::ndarray Y) {
  if (X.get_nd() != 1 || Y.get_nd() != 1) {
    throw std::runtime_error("X and Y should be 1-dimentional array");
  }

  if (X.shape(0) != Y.shape(0)) {
    throw std::runtime_error("X and Y should be same size");
  }

  unsigned long N = X.shape(0);
  float *sims = new float[N];

  std::vector< std::set < std::pair<wchar_t, wchar_t> > > _X(N), _Y(N);

    wchar_t buf[1024];
//  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
//    Py_ssize_t copied =
    PyUnicode_AsWideChar(reinterpret_cast<PyObject **>(X.get_data())[i], buf, 1024);
    wchar_t * c = buf;
    if (c[0] != 0) {
      _X[i].insert(std::make_pair(0, c[0]));
      for (; *c != 0; ++c) {
        _X[i].insert(std::make_pair(c[0], c[1]));
      }
      _X[i].insert(std::make_pair(c[-1], c[0]));
    }
  }

//  #pragma omp parallel for
  for (int j = 0; j < N; j++) {
//    Py_ssize_t copied =
    PyUnicode_AsWideChar(reinterpret_cast<PyObject **>(Y.get_data())[j], buf, 1024);
    wchar_t * c = buf;
    if (c[0] != 0) {
      _Y[j].insert(std::make_pair(0, c[0]));
      for (; *c != 0; ++c) {
        _Y[j].insert(std::make_pair(c[0], c[1]));
      }
      _Y[j].insert(std::make_pair(c[-1], c[0]));
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
      sims[i] = similarity_measure::jaccard::get_sim_score(_X[i], _Y[i]);
  }

  return np::from_data(sims, np::dtype::get_builtin<float>(),
    p::make_tuple(N), p::make_tuple(sizeof(float)), p::object());
}