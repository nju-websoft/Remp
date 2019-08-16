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
  struct WhitespaceTokenizer {
    using token_type = typename std::basic_string<char_type>;

    template<class input_type, class container_type>
    inline bool operator()(input_type * c, container_type & grams) const {
      using container_token_type = typename container_type::value_type;
      for (auto i = c; *i != 0; ++i) {
        if (!is_whitespace(*i)) {
          auto s = i;
          while (*i != 0 && !is_whitespace(*i)) ++i;
          grams.insert(container_token_type{s, i});
        }
      }

      return true;
    }
  };
}

#endif