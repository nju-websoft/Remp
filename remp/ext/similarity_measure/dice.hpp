#ifndef REMP_EXT_SIMILARITY_MEASURE_DICE_HPP
#define REMP_EXT_SIMILARITY_MEASURE_DICE_HPP

#pragma once

#include <algorithm>
#include "../util/counter_iterator.hpp"

namespace similarity_measure {
  namespace dice {
    template<class container_type>
    inline float get_raw_score(const container_type & x, const container_type & y) {
      using char_type = typename container_type::value_type;
      if (x.size() == 0 || y.size() == 0) {
        return 0.0f;
      }

      __REMP_EXT_impl::counter_iterator<container_type> i;
      i = std::set_intersection(std::begin(x), std::end(x), std::begin(y), std::end(y), i);
      return i.count() * 1.0 / (x.size() + y.size());
    }

    template<class container_type>
    inline float get_sim_score(const container_type & x, const container_type & y) {
      return get_raw_score<container_type>(x, y);
    }
  }
}

#endif