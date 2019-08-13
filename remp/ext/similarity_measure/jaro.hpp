#ifndef REMP_EXT_SIMILARITY_MEASURE_JARO_HPP
#define REMP_EXT_SIMILARITY_MEASURE_JARO_HPP

#pragma once

#include <vector>

namespace similarity_measure {
  namespace jaro {
    template<class container_type>
    inline float get_raw_score(const container_type & x, const container_type & y) {
      auto lx = x.size(), ly = y.size();
      if (lx == 0 || ly == 0) {
        return 0.0f;
      }

      auto max_len = lx >= ly ? lx : ly;
      auto search_range = max_len / 2 - 1;

      std::vector<std::size_t> flags_s1(lx, 0), flags_s2(ly, 0);

      std::size_t common_chars = 0, low = 0, high = 0, i = 0, j = 0;
      // Finding the number of common characters in two strings
      for (i = 0; i < lx; i++) {
        low = i > search_range ? i - search_range : 0;
        high = i + search_range < ly ? i + search_range : ly - 1;
        for (j = low; j <= high; j++) {
          if (flags_s2[j] == 0 && y[j] == x[i]) {
              flags_s1[i] = flags_s2[j] = 1;
              common_chars += 1;
              break;
          }
        }
      }

      if (common_chars == 0) {
        return 0.0f;
      }

      std::size_t trans_count = 0, k = 0;

      // Finding the number of transpositions and Jaro distance
      for (i = 0; i < lx; i++) {
        if (flags_s1[i] == 1) {
          for (j = k; j < ly; j++) {
            if (flags_s2[j] == 1) {
              k = j + 1;
              break;
            }
          }
          if (x[i] != y[j]) {
            trans_count += 1;
          }
        }
      }

      trans_count /= 2;

      float ccs = common_chars;

      return (ccs / lx + ccs / ly + (ccs - trans_count) / ccs) / 3;
    }

    template<class container_type>
    inline float get_sim_score(const container_type & x, const container_type & y) {
      return get_raw_score<container_type>(x, y);
    }
  }
}

#endif