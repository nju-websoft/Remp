#ifndef REMP_EXT_UTIL_COUNTER_ITERATOR_HPP
#define REMP_EXT_UTIL_COUNTER_ITERATOR_HPP

#pragma once

#include <iterator>

namespace __REMP_EXT_impl {
  template <class container_type>
  class counter_iterator
    : public std::iterator<std::output_iterator_tag, typename container_type::value_type> {
  public:
    template<class T>
    struct copy_assign_blackhole {
    public:
      inline copy_assign_blackhole & operator=(const T &) {
        return *this;
      }
    };

  public:
    counter_iterator() : count_(0) {}

    ~counter_iterator() {}

    inline counter_iterator<container_type> & operator=(const counter_iterator<container_type> & i) {
      count_ = i.count_;
      return *this;
    }

    inline counter_iterator<container_type> & operator++() {
      ++count_;
      return *this;
    }

    inline copy_assign_blackhole<typename container_type::value_type> operator*() const {
      return copy_assign_blackhole<typename container_type::value_type> {};
    }

    inline unsigned long count() const {
      return count_;
    }
  private:
    unsigned long count_;
  };
}

#endif