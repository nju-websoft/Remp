#ifndef REMP_EXT_SIMILARITY_MEASURE_JACCARD_HPP
#define REMP_EXT_SIMILARITY_MEASURE_JACCARD_HPP

#pragma once

#include <algorithm>
#include <set>
#include "../util/counter_iterator.hpp"

namespace similarity_measure {
  template<class token_type>
  struct Jaccard {
    using container_type = std::set<token_type>;

    template<class _container_type>
    inline float get_raw_score(const _container_type & x, const _container_type & y) {
      if (x.size() == 0 || y.size() == 0) {
        return 0.0f;
      }

      __REMP_EXT_impl::counter_iterator< container_type > i, u;
      i = std::set_intersection(std::begin(x), std::end(x), std::begin(y), std::end(y), i);
      u = std::set_union(std::begin(x), std::end(x), std::begin(y), std::end(y), u);
      return i.count() * 1.0 / u.count();
    }

    template<class _container_type>
    inline float get_sim_score(const _container_type & x, const _container_type & y) {
      return get_raw_score(x, y);
    }
  };
}

#endif