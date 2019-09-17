#ifndef REMP_EXT_SIMILARITY_MEASURE_COSINE_HPP
#define REMP_EXT_SIMILARITY_MEASURE_COSINE_HPP

#pragma once

#include <algorithm>
#include <set>
#include <cmath>
#include "../util/counter_iterator.hpp"

namespace similarity_measure {
  template<class token_type>
  struct Cosine {
    using container_type = std::set<token_type>;

    template<class _container_type>
    inline float get_raw_score(const _container_type & x, const _container_type & y) {
      if (x.size() == 0 || y.size() == 0) {
        return 0.0f;
      }

      __REMP_EXT_impl::counter_iterator< container_type > i;
      i = std::set_intersection(std::begin(x), std::end(x), std::begin(y), std::end(y), i);
      return i.count() * 1.0f / std::sqrt(x.size() * y.size() * 1.0f);
    }

    template<class _container_type>
    inline float get_sim_score(const _container_type & x, const _container_type & y) {
      return get_raw_score(x, y);
    }
  };
}

#endif