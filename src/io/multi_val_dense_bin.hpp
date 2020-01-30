/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_


#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

namespace LightGBM {


template <typename VAL_T>
class MultiValDenseBin : public MultiValBin {
public:

  explicit MultiValDenseBin(data_size_t num_data, int num_bin, int num_feature)
    : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature) {
    data_.size(static_cast<size_t>(num_data_) * num_feature_, 0);
  }

  ~MultiValDenseBin() {
  }

  data_size_t num_data() const override {
    return num_data_;
  }

  int num_bin() const override {
    return num_bin_;
  }


  void PushOneRow(int , data_size_t idx, const std::vector<uint32_t>& values) override {
    auto start = RowPtr(idx);
    auto end = RowPtr(idx + 1);
    CHECK(num_feature_ == static_cast<int>(values.size()));
    for (auto i = start; i < end; ++i) {
      data_[i] = values[i - start];
    }
  }

  void FinishLoad() override {

  }

  bool IsSparse() override{
    return false;
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
    }
  }

  #define ACC_GH(hist, i, g, h) \
  const auto ti = static_cast<int>(i) << 1; \
  hist[ti] += g; \
  hist[ti + 1] += h; \

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(gradients + data_indices[i + prefetch_size]);
        PREFETCH_T0(hessians + data_indices[i + prefetch_size]);
        PREFETCH_T0(data_.data() + RowPtr(data_indices[i + prefetch_size]));
      }
      for (int64_t idx = RowPtr(data_indices[i]); idx < RowPtr(data_indices[i] + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[data_indices[i]], hessians[data_indices[i]]);
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(gradients + i + prefetch_size);
        PREFETCH_T0(hessians + i + prefetch_size);
        PREFETCH_T0(data_.data() + RowPtr(i + prefetch_size));
      }
      for (int64_t idx = RowPtr(i); idx < RowPtr(i + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[i], hessians[i]);
      }
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(gradients + data_indices[i + prefetch_size]);
        PREFETCH_T0(data_.data() + RowPtr(data_indices[i + prefetch_size]));
      }
      for (int64_t idx = RowPtr(data_indices[i]); idx < RowPtr(data_indices[i] + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[data_indices[i]], 1.0f);
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(gradients + i + prefetch_size);
        PREFETCH_T0(data_.data() + RowPtr(i + prefetch_size));
      }
      for (int64_t idx = RowPtr(i); idx < RowPtr(i + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[i], 1.0f);
      }
    }
  }
  #undef ACC_GH

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    data_.clear();
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      for (int64_t j = other_bin->RowPtr(used_indices[i]); j < other_bin->RowPtr(used_indices[i] + 1); ++j) {
        data_.push_back(other_bin->data_[j]);
      }
    }
  }

  inline int64_t RowPtr(data_size_t idx) const {
    return static_cast<int64_t>(idx)* num_feature_;
  }

  MultiValDenseBin<VAL_T>* Clone() override;

private:
  data_size_t num_data_;
  int num_bin_;
  int num_feature_;
  std::vector<VAL_T> data_;

  MultiValDenseBin<VAL_T>(const MultiValDenseBin<VAL_T>& other)
    : num_data_(other.num_data_), data_(other.data_), num_feature_(other.num_feature_){
  }
};

template<typename VAL_T>
MultiValDenseBin<VAL_T>* MultiValDenseBin<VAL_T>::Clone() {
  return new MultiValDenseBin<VAL_T>(*this);
}



}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
