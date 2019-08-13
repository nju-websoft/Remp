#ifndef REMP_EXT_TOKENIZER_QGRAM_TOKENIZER_HPP
#define REMP_EXT_TOKENIZER_QGRAM_TOKENIZER_HPP

#pragma once

#include <array>
#include <set>
#include <algorithm>

namespace tokenizer {
  template<typename char_type, int qval>
  class QgramTokenizer {
  public:
    using token_type = typename std::array<wchar_t, qval>;
    static bool tokenize(char_type * begin, std::set< token_type > & grams) {
      token_type gram;
      std::array<char_type, 2 * qval - 2> buf;
      buf.fill(0);

      auto output_it = std::begin(buf) + qval - 1, output_end = std::end(buf), input_end = std::end(buf);
      auto i = begin;

      for (; *i != 0 && output_it != output_end; ++i, ++output_it) {
        *output_it = *i;
        std::copy(output_it - qval + 1, output_it + 1, std::begin(gram));
        grams.insert(gram);
      }

      for (; output_it < output_end; ++output_it) {
        std::copy(output_it - qval + 1, output_it + 1, std::begin(gram));
        grams.insert(gram);
      }

      if (*i != 0) {
        // length of string >= N
        for (i = begin; i[qval] != 0; ++i) {
          std::copy(i, i + qval, std::begin(gram));
          grams.insert(gram);
        }
        std::copy(i, i + qval, std::begin(gram));
        grams.insert(gram);
        buf.fill(0);
        std::copy(i + 1, i + qval, std::begin(buf));
        input_end = std::begin(buf) + qval - 1;
      } else {
        buf.fill(0);
        std::copy(begin, i, std::begin(buf));
        input_end = std::begin(buf) + (i - begin);
      }

      for (auto input_it = std::begin(buf); input_it != input_end; ++input_it) {
        std::copy(input_it, input_it + qval, std::begin(gram));
        grams.insert(gram);
      }
      
      return true;
    }
  };

  template<typename char_type>
  class QgramTokenizer<char_type, 2> {
  public:
    using token_type = typename std::pair<wchar_t, wchar_t>;

    static bool tokenize(char_type * c, std::set< token_type > & grams) {
      if (c[0] != 0) {
        grams.insert(std::make_pair(0, c[0]));
        for (; *c != 0; ++c) {
          grams.insert(std::make_pair(c[0], c[1]));
        }
        grams.insert(std::make_pair(c[-1], c[0]));
      }

      return true;
    }
  };
}

#endif