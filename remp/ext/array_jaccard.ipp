template<class tokenizer>
np::ndarray array_jaccard(np::ndarray X, np::ndarray Y) {
  if (X.get_nd() != 1 || Y.get_nd() != 1) {
    throw std::runtime_error("X and Y should be 1-dimentional array");
  }

  if (X.shape(0) != Y.shape(0)) {
    throw std::runtime_error("X and Y should be same size");
  }

  unsigned long N = X.shape(0);
  float *sims = new float[N];

  //using tokenizer = tokenizer::QgramTokenizer<wchar_t, qval>;

  std::vector< std::set < typename tokenizer::token_type > > _X(N), _Y(N);

    wchar_t buf[1024];
//  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
//    Py_ssize_t copied =
    PyUnicode_AsWideChar(reinterpret_cast<PyObject **>(X.get_data())[i], buf, 1024);
    tokenizer::tokenize(buf, _X[i]);
  }

//  #pragma omp parallel for
  for (int j = 0; j < N; j++) {
//    Py_ssize_t copied =
    PyUnicode_AsWideChar(reinterpret_cast<PyObject **>(Y.get_data())[j], buf, 1024);
    tokenizer::tokenize(buf, _Y[j]);
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
      sims[i] = similarity_measure::jaccard::get_sim_score(_X[i], _Y[i]);
  }

  return np::from_data(sims, np::dtype::get_builtin<float>(),
    p::make_tuple(N), p::make_tuple(sizeof(float)), p::object());
}