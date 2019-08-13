#ifndef REMP_EXT_UTIL_QGRAMS_HPP
#define REMP_EXT_UTIL_QGRAMS_HPP

#include <array>
#include <set>
#include <string>

namespace __REMP_EXT_impl {
  template <int N, typename char_type=wchar_t>
  void qgrams(std::set < std::array<char_type, N> > &grams, const char_type * begin) {
    std::array<char_type, N> gram;
    std::array<char_type, 2 * N - 2> buf;
    buf.fill(0);

    auto output_it = std::begin(buf) + N - 1, output_end = std::end(buf), input_end = std::end(buf);
    auto i = begin;

    for (; *i != 0 && output_it != output_end; ++i, ++output_it) {
      *output_it = *i;
      std::copy(output_it - N + 1, output_it + 1, std::begin(gram));
      grams.insert(gram);
    }

    for (; output_it < output_end; ++output_it) {
      std::copy(output_it - N + 1, output_it + 1, std::begin(gram));
      grams.insert(gram);
    }

    if (*i != 0) {
      // length of string >= N
      for (i = begin; i[N] != 0; ++i) {
        std::copy(i, i + N, std::begin(gram));
        grams.insert(gram);
      }
      std::copy(i, i + N, std::begin(gram));
      grams.insert(gram);
      buf.fill(0);
      std::copy(i + 1, i + N, std::begin(buf));
      input_end = std::begin(buf) + N - 1;
    } else {
      buf.fill(0);
      std::copy(begin, i, std::begin(buf));
      input_end = std::begin(buf) + (i - begin);
    }

    for (auto input_it = std::begin(buf); input_it != input_end; ++input_it) {
      std::copy(input_it, input_it + N, std::begin(gram));
      grams.insert(gram);
    }
  }
}

#endif