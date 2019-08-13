#ifndef REMP_EXT_SIMILARITY_MEASURE_GENERALIZED_JACCARD_HPP
#define REMP_EXT_SIMILARITY_MEASURE_GENERALIZED_JACCARD_HPP

#pragma once

#include <algorithm>
#include <tuple>
#include <vector>
#include <type_traits>
#include <iterator>

namespace __REMP_EXT_impl {
  template <class T>
  struct remove_iterator
  {
    typedef typename T::value_type type;
  };

  template<class T>
  struct remove_iterator<T*>
  {
    typedef T type;
  };
}

namespace similarity_measure {
  namespace generalized_jaccard {
    template<typename index_iterator_type, typename sim_type>
    inline float get_raw_score(
      index_iterator_type xbegin, index_iterator_type xend, 
      index_iterator_type ybegin, index_iterator_type yend, 
      const sim_type * base_sims,
      std::size_t width, float threshold
      ){

      if (xbegin == xend || ybegin == yend) {
        return 0.0f;
      }

      std::vector< std::tuple<int, int, sim_type> > scores;
      scores.reserve((xend - xbegin) * (yend - ybegin) / 2);

      for (auto xi = xbegin; xi < xend; ++xi) {
        for (auto yi = ybegin; yi < yend; ++yi) {
          auto sim = *(base_sims + *xi * width + *yi);
          if (sim > threshold) {
            scores.emplace_back(xi - xbegin, yi - ybegin, sim);
          }
        }
      }

      std::sort(std::begin(scores), std::end(scores), [](auto x, auto y) {
        return std::get<2>(x) > std::get<2>(y);
      });

      std::set< int > set_x, set_y;

      std::size_t match_count = 0;
      float match_score = 0.0f;

      for (const auto & t : scores) {
        if (set_x.find(std::get<0>(t)) == set_x.end()
        && set_y.find(std::get<1>(t)) == set_y.end()) {
          set_x.insert(std::get<0>(t));
          set_y.insert(std::get<1>(t));
          match_score += std::get<2>(t);
          match_count += 1;
        }
      }

      return match_score / ((xend - xbegin) + (yend - ybegin) - match_count);
    }

    template<typename index_iterator_type, typename sim_type>
    inline float get_sim_score(
      index_iterator_type xbegin, index_iterator_type xend, 
      index_iterator_type ybegin, index_iterator_type yend, 
      const sim_type * base_sims,
      std::size_t width, float threshold
      ){
      return get_raw_score(xbegin, xend, ybegin, yend, base_sims, width, threshold);
    }
  }
}

#endif