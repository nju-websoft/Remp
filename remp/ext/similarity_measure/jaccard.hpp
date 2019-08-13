#ifndef REMP_EXT_SIMILARITY_MEASURE_JACCARD_HPP
#define REMP_EXT_SIMILARITY_MEASURE_JACCARD_HPP

#pragma once

#include <algorithm>
#include "../util/counter_iterator.hpp"

namespace similarity_measure {
  namespace jaccard {
    template<class container_type>
    inline float get_raw_score(const container_type & x, const container_type & y) {
      if (x.size() == 0 || y.size() == 0) {
        return 0.0f;
      }

      __REMP_EXT_impl::counter_iterator<container_type> i, u;
      i = std::set_intersection(std::begin(x), std::end(x), std::begin(y), std::end(y), i);
      u = std::set_union(std::begin(x), std::end(x), std::begin(y), std::end(y), u);
      return i.count() * 1.0 / u.count();
    }

    template<class container_type>
    inline float get_sim_score(const container_type & x, const container_type & y) {
      return get_raw_score<container_type>(x, y);
    }
  }
}

#endif