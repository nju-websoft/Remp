#ifndef REMP_EXT_SIMILARITY_JOIN_JACCARD_JOIN_HPP
#define REMP_EXT_SIMILARITY_JOIN_JACCARD_JOIN_HPP

#include <boost/python/numpy.hpp>

#include <vector>
#include <deque>
#include <tuple>
#include <cmath>
#include <map>

#include <omp.h>

#include "../similarity_measure/jaccard.hpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace similarity_join {
  namespace __impl {
    template<typename T>
    void build_length_inverted_index(const std::vector< std::set <T> > & items, std::vector< std::deque<std::size_t> > &length_index) {
      length_index.clear();
      for (std::size_t i = 0, il = items.size(); i < il; i++) {
        std::size_t length = items[i].size();
        while (length_index.size() <= length) {
          length_index.emplace_back();
        }
        length_index[length].push_back(i);
      }
    }
    template<typename T>
    void build_character_weight_index(const std::vector< std::set <T> > & items, std::map< T, std::size_t > &character_index) {
      character_index.clear();
      for (std::size_t i = 0, il = items.size(); i < il; i++) {
        for (const T ch : items[i]) {
          if (character_index.find(ch) == character_index.end()) {
            character_index[ch] = 1;
          } else {
            ++character_index[ch];
          }
        }
      }
    }
  }

  using result_triple = std::tuple<std::size_t, std::size_t, float>;

  template<typename T>
  boost::python::dict jaccard_join(const std::vector< std::set <T> > & tokens_1, const std::vector< std::set <T> > & tokens_2, float threshold) {
    boost::python::dict result;
    std::vector< std::deque<std::size_t> > length_index_1, length_index_2;
    std::map<T, std::size_t> character_index_1, character_index_2;
    #pragma omp parallel
    {
      int ID = omp_get_thread_num(), NTHREADS = omp_get_max_threads();
      if (NTHREADS < 4) {
        __impl::build_length_inverted_index(tokens_1, length_index_1);
        __impl::build_length_inverted_index(tokens_2, length_index_2);
        __impl::build_character_weight_index(tokens_1, character_index_1);
        __impl::build_character_weight_index(tokens_2, character_index_2);
      } else {
        if (ID == 0) {
          __impl::build_length_inverted_index(tokens_1, length_index_1);
        } else if (ID == 1) {
          __impl::build_length_inverted_index(tokens_2, length_index_2);
        } else if (ID == 2) {
          __impl::build_character_weight_index(tokens_1, character_index_1);
        } else if (ID == 3) {
          __impl::build_character_weight_index(tokens_2, character_index_2);
        }
      }
    }

    std::size_t nthreads = omp_get_max_threads();
    std::vector< std::deque< result_triple > > buckets { nthreads, std::deque< result_triple > {} };

    for (std::size_t length_1 = 1; length_1 < length_index_1.size(); length_1++) {
      std::size_t min_length_2 = std::max(1, (int)(length_1 * threshold));
      std::size_t max_length_2 = std::min((int)length_index_2.size() - 1, (int)(length_1 / threshold));
      for (std::size_t length_2 = min_length_2; length_2 <= max_length_2; length_2++) {
        auto & group_1 = length_index_1[length_1];
        auto & group_2 = length_index_2[length_2];
        auto length_threhold = (std::size_t)std::ceil((length_1 + length_2) / (1.f + 1.f / threshold));
        std::size_t total = group_1.size() * group_2.size();
        if (total == 0) continue;
        std::size_t M = group_2.size();
        #pragma omp parallel
        {
          std::size_t ID, istart, iend, i, NTHREADS;
          ID = omp_get_thread_num();
          NTHREADS = omp_get_max_threads();
          istart = ID * total / NTHREADS;
          iend = (ID + 1) * total / NTHREADS;
          if (ID == nthreads - 1) iend = total;
          std::size_t idx_1 = istart / M, idx_2 = istart % M;
          for (i = istart; i < iend; i++) {
            auto sim = ::similarity_measure::jaccard::get_sim_score(tokens_1[group_1[idx_1]], tokens_2[group_2[idx_2]]);
            if (sim >= threshold) {
              buckets[ID].emplace_back(group_1[idx_1], group_2[idx_2], sim);
            }
            idx_2 += 1;
            if (idx_2 == M) {
              idx_1 += 1;
              idx_2 = 0;
            }
          }
        }
      }
    }

    std::size_t count = 0;
    for (int i = 0; i < nthreads; i++) {
      count += buckets[i].size();
    }

    unsigned long * idx_1 = new unsigned long[count];
    unsigned long * idx_2 = new unsigned long[count];
    float *sims = new float[count];
    count = 0;
    for (int i = 0; i < nthreads; i++) {
      for (int j = 0; j < buckets[i].size(); j++) {
        idx_1[count] = std::get<0>(buckets[i][j]);
        idx_2[count] = std::get<1>(buckets[i][j]);
        sims[count] = std::get<2>(buckets[i][j]);
        ++count;
      }
    }
    result["idx_1"] = np::from_data(idx_1, np::dtype::get_builtin<unsigned long>(),
        p::make_tuple(count), p::make_tuple(sizeof(unsigned long)), p::object());
    result["idx_2"] = np::from_data(idx_2, np::dtype::get_builtin<unsigned long>(),
        p::make_tuple(count), p::make_tuple(sizeof(unsigned long)), p::object());
    result["sim"] = np::from_data(sims, np::dtype::get_builtin<float>(),
        p::make_tuple(count), p::make_tuple(sizeof(float)), p::object());

    return result;
  }
}

#endif
