#include <set>
#include <wchar.h>

template<class tokenizer, class container_type=std::set<typename tokenizer::token_type> >
np::ndarray array_jaccard(np::ndarray X, np::ndarray Y) {
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

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    container_type x, y;
    switch (PyUnicode_KIND(_X[i])) {
      case 2:
        tokenizer::tokenize((wchar_t *)PyUnicode_2BYTE_DATA(_X[i]), x);
        break;
      case 1:
        tokenizer::tokenize((const char *)PyUnicode_1BYTE_DATA(_X[i]), x);
        break;
      default:
        throw std::runtime_error("string fromat error");
    }

    switch (PyUnicode_KIND(_Y[i])) {
      case 2:
        tokenizer::tokenize((wchar_t *)PyUnicode_2BYTE_DATA(_Y[i]), y);
        break;
      case 1:
        tokenizer::tokenize((const char *)PyUnicode_1BYTE_DATA(_Y[i]), y);
        break;
      default:
        throw std::runtime_error("string fromat error");
    }

    sims[i] = similarity_measure::jaccard::get_sim_score(x, y);
  }

  return np::from_data(sims, np::dtype::get_builtin<float>(),
    p::make_tuple(N), p::make_tuple(sizeof(float)), p::object());
}