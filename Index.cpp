/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexFlat.h"
#include "FaissAssert.h"

#include <cstring>

namespace faiss {

Index::~Index ()
{
}


void Index::train(idx_t /*n*/, const float* /*x*/) {
    // does nothing by default
}


void Index::range_search (idx_t , const float *, float,
                          RangeSearchResult *) const
{
  FAISS_THROW_MSG ("range search not implemented");
}

void Index::assign (idx_t n, const float * x, idx_t * labels, idx_t k)
{
  float * distances = new float[n * k];
  ScopeDeleter<float> del(distances);
  search (n, x, k, distances, labels);
}
int  Index::reserve(faiss::Index::idx_t n){
   printf("\nNOTICE:reserve not implemented for this index \n");
}
void Index::add_with_ids(
    idx_t /*n*/,
    const float* /*x*/,
    const long* /*xids*/) {
  FAISS_THROW_MSG ("add_with_ids not implemented for this type of index");
}

long Index::remove_ids(const IDSelector& /*sel*/) {
  FAISS_THROW_MSG ("remove_ids not implemented for this type of index");
  return -1;
}

long Index::remove_ids(const idx_t & /*sel*/) {
  FAISS_THROW_MSG ("remove_ids not implemented for this type of index");
  return -1;
}


void Index::reconstruct (idx_t, float * ) const {
  FAISS_THROW_MSG ("reconstruct not implemented for this type of index");
}

void Index::update (idx_t key,const float * new_f) const {
  FAISS_THROW_MSG ("reconstruct not implemented for this type of index");
}


void Index::reconstruct_n (idx_t i0, idx_t ni, float *recons) const {
  for (idx_t i = 0; i < ni; i++) {
    reconstruct (i0 + i, recons + i * d);
  }
}


void Index::search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                    float *distances, idx_t *labels,
                                    float *recons) const {
  search (n, x, k, distances, labels);
  for (idx_t i = 0; i < n; ++i) {
    for (idx_t j = 0; j < k; ++j) {
      idx_t ij = i * k + j;
      idx_t key = labels[ij];
      float* reconstructed = recons + ij * d;
      if (key < 0) {
        // Fill with NaNs
        memset(reconstructed, -1, sizeof(*reconstructed) * d);
      } else {
        reconstruct (key, reconstructed);
      }
    }
  }
}


void Index::compute_residual (const float * x,
                              float * residual, idx_t key) const {
  reconstruct (key, residual);
  for (size_t i = 0; i < d; i++)
    residual[i] = x[i] - residual[i];
}


void Index::display () const {
  printf ("Index: %s  -> %ld elements\n", typeid (*this).name(), ntotal);
}

ErrorTypes Index::get_error_state() {
  auto ret = error_state;
  error_state = Faiss_Error_OK;
  return ret;
}

void Index::set_max_size(size_t new_size)
{
  max_size = new_size;
}

size_t Index::get_max_size() const
{
  return max_size;
}

void Index::set_user_reserve(bool v)
{
   index_user_reserve = v;
}

void Index::set_use_int8_norms(bool flag)
{
  use_int8_norms = flag;
}

bool Index::get_use_int8_norms()
{
  return use_int8_norms;
}

}

