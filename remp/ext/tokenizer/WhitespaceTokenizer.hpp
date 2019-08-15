#ifndef REMP_EXT_TOKENIZER_WHITESPACE_TOKENIZER_HPP
#define REMP_EXT_TOKENIZER_WHITESPACE_TOKENIZER_HPP

#pragma once

#include <array>
#include <set>
#include <algorithm>
#include <string>

namespace tokenizer {
  template<typename char_type>
  inline bool is_whitespace(char_type ch) {
    return (ch == (char_type)' ' || ch == (char_type)'\t' || ch == (char_type)'\n');
  }

  template<typename char_type>
  class WhitespaceTokenizer {
  public:
    using token_type = typename std::basic_string<char_type>;

    template<class container_type>
    static bool tokenize(char_type * c, container_type & grams) {
      for (auto i = c; *i != 0; ++i) {
        if (!is_whitespace(*i)) {
          auto s = i;
          while (*i != 0 && !is_whitespace(*i)) ++i;
          grams.insert(token_type{s, i - s});
        }
      }

      return true;
    }

    template<class container_type>
    static bool tokenize(const char * c, container_type & grams) {
      for (auto i = c; *i != 0; ++i) {
        if (!is_whitespace(*i)) {
          auto s = i;
          while (*i != 0 && !is_whitespace(*i)) ++i;
          typename container_type::value_type part;
          part.reserve(i - s);
          for (int j = 0; j < i - s; j++) {
            part.push_back(*(i + j));
          }
          grams.insert(part);
        }
      }

      return true;
    }

  };
}

#endif