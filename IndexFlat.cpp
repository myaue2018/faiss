/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved

#include "IndexFlat.h"

#include <cstring>
#include <fcntl.h>
#include <zconf.h>
#include <iostream>
#include <glog/logging.h>
#include <tbb/task_group.h>
#include "utils.h"
#include "Heap.h"

#include "FaissAssert.h"

#include "AuxIndexStructures.h"

namespace faiss {

static const size_t BLK_SIZE = 65536;

IndexFlat::IndexFlat (idx_t d, MetricType metric, DataType data_type):
            Index(d, metric, data_type), xb_int8(BLK_SIZE * d)
{
}



void IndexFlat::add (idx_t n, const float *x) {
    if (data_type == DATA_IINT8) {
        uint8_t* x_ = new uint8_t[n * d];
        FloatToUint8(x_, x, n * d);
        xb_int8.append(x_, n * d);
        delete [] x_;
    } else {
        xb.insert(xb.end(), x, x + n * d);
    }
    ntotal += n;
}


void IndexFlat::add (idx_t n, const uint8_t *x) {
    xb_int8.append(x, n * d);
    ntotal += n;
}


void IndexFlat::reset() {
    xb.clear();
    xb_int8.clear();
    ntotal = 0;
}


void IndexFlat::search (idx_t n, const float *x, idx_t k,
                               float *distances, idx_t *labels) const
{
    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        if (data_type == DATA_IINT8) {
            queryNorms.resize(n);
            knn_inner_product (x, xb_int8, d, n, &res, queryNorms.data(), index_int8_cosine_ignore_negative);
        } else {
            knn_inner_product (x, xb.data(), d, n, ntotal, &res);
        }
    } else if (metric_type == METRIC_L2) {
        FAISS_THROW_IF_NOT(data_type == DATA_IFLOAT);
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        knn_L2sqr (x, xb.data(), d, n, ntotal, &res);
    }
}

void IndexFlat::range_search (idx_t n, const float *x, float radius,
                              RangeSearchResult *result) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product (x, xb.data(), d, n, ntotal,
                                        radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr (x, xb.data(), d, n, ntotal, radius, result);
            break;
    }
}


void IndexFlat::compute_distance_subset (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            const idx_t *labels) const
{
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx (
                 distances,
                 x, xb.data(), labels, d, n, k);
            break;
    }

}

long IndexFlat::remove_ids (const IDSelector & sel)
{
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member (i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove (&xb[d * j], &xb[d * i], sizeof(xb[0]) * d);
            }
            j++;
        }
    }
    long nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize (ntotal * d);
    }
    return nremove;
}



void IndexFlat::reconstruct (idx_t key, float * recons) const
{
    memcpy (recons, &(xb[key * d]), sizeof(*recons) * d);
}

long IndexFlat::remove_ids(const idx_t &idx) {

    if(idx >= ntotal){
        return -1;
    }

    if(ntotal > 0){
        if (data_type == DATA_IINT8) {
            xb_int8.replace(idx * d, &xb_int8[(ntotal - 1) * d], sizeof(uint8_t) * d);
            ntotal -= 1;
            xb_int8.resize(ntotal * d);
        } else {
            memcpy(&xb[d * idx],&xb[d * (ntotal-1)],sizeof(float)*d);
            ntotal -= 1;
            xb.resize(ntotal * d);
        }
    }

    return 1;
}

int IndexFlat::reserve(faiss::Index::idx_t n) {
    if (data_type == DATA_IINT8) {
        xb_int8.reserve(n*d);
    } else {
        xb.reserve(n*d);
    }
    return 1;
}

void IndexFlat::update(idx_t key, const float *recons) const {
    if(key>=ntotal){
        return;
    }
    if (data_type == DATA_IINT8) {
        uint8_t* recons_ = new uint8_t[d];
        FloatToUint8(recons_, recons, d);
        xb_int8.replace(key * d, recons_, sizeof(uint8_t) * d);
        delete [] recons_;
    } else {
        memcpy((float*)(&xb[d * key]),(float*)recons,sizeof(float)*d);
    }
}

void IndexFlat::get_query_norms(float *query_norms)
{
    memcpy(query_norms, queryNorms.data(), queryNorms.size() * sizeof(float));
}

void IndexFlat::get_feature_norms(idx_t n, idx_t k, const idx_t *ids, float *feature_norms)
{
    for (size_t i = 0; i < n * k; ++i) {
        feature_norms[i] = fvec_norm_L2r_ref_uint8(&xb_int8[ids[i] * d], d);
    }
}

bool IndexFlat::load_index_from_file(const std::string& index_file) {
    // open file and get index size
    FILE *pFile = fopen(index_file.c_str(), "rb");
    if (pFile == nullptr) {
        LOG(ERROR) << "[load_index_from_file] open file error!";
        return false;
    }
    if (fseek(pFile, 0L, SEEK_END) != 0) {
        LOG(ERROR) << "[load_index_from_file] seek for file end error!";
        fclose(pFile);
        return false;
    }
    long index_size = ftell(pFile);
    rewind(pFile);
    index_size = index_size / sizeof(int8_t);
    bool ret = true;
    // load index data
//    tbb::task_group group;
//    group.run([&]{
        long batch_size = BLK_SIZE * d; // shrink the data I/O size to 1/8 of block size
        long left_size = index_size;
        xb_int8.resize(size_t(index_size));
        while (left_size > 0) {
            long count_in_batch = left_size > batch_size ? batch_size : left_size;
            auto cnt = fread((char *) &xb_int8[index_size - left_size], sizeof(char), count_in_batch * sizeof(int8_t), pFile);
            if (cnt != count_in_batch * sizeof(int8_t)) {
                LOG(ERROR) << "[load_index_from_file] read file error!";
                fclose(pFile);
                ret = false;
                break;
            }
            ntotal += count_in_batch / d;
            left_size -= count_in_batch;
        }
        fclose(pFile);
//    });
//    group.wait();
    return ret;
}

bool IndexFlat::save_index_to_file(const std::string& index_file) {
    // open file for write
    if (ntotal == 0) {
        return false;
    }
    FILE *pFile = fopen(index_file.c_str(), "wb");
    if (pFile == nullptr) {
        return false;
    }
    bool ret = true;
    // save index data
//    tbb::task_group group;
//    group.run([&]{
        size_t batch_size = BLK_SIZE * d;   // shrink the data I/O size to 1/8 of block size
        auto left_size = xb_int8.size();
        std::cout << "notatol & xb_.size(): " << ntotal << " " << left_size << std::endl;
        size_t w_cnt = 0;
        while (left_size > 0) {
            size_t count_in_batch = left_size > batch_size ? batch_size : left_size;
            auto cnt = fwrite((char *) &xb_int8[xb_int8.size() - left_size], sizeof(char), count_in_batch * sizeof(int8_t), pFile);
            w_cnt += cnt;
            if (cnt != count_in_batch * sizeof(int8_t)) {
                LOG(ERROR) << "[save_index_to_file] write file error!";
                ret = false;
                break;
            }
            left_size -= count_in_batch;
        }
        fclose(pFile);
        std::cout << "write bytes count: " << w_cnt << std::endl;
//    });
//    group.wait();
    return ret;
}

/***************************************************
 * IndexFlatL2BaseShift
 ***************************************************/

IndexFlatL2BaseShift::IndexFlatL2BaseShift (idx_t d, size_t nshift, const float *shift):
    IndexFlatL2 (d), shift (nshift)
{
    memcpy (this->shift.data(), shift, sizeof(float) * nshift);
}

void IndexFlatL2BaseShift::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT (shift.size() == ntotal);

    float_maxheap_array_t res = {
        size_t(n), size_t(k), labels, distances};
    knn_L2sqr_base_shift (x, xb.data(), d, n, ntotal, &res, shift.data());
}



/***************************************************
 * IndexRefineFlat
 ***************************************************/

IndexRefineFlat::IndexRefineFlat (Index *base_index):
    Index (base_index->d, base_index->metric_type),
    refine_index (base_index->d, base_index->metric_type),
    base_index (base_index), own_fields (false),
    k_factor (1)
{
    is_trained = base_index->is_trained;
    FAISS_THROW_IF_NOT_MSG (base_index->ntotal == 0,
                      "base_index should be empty in the beginning");
}

IndexRefineFlat::IndexRefineFlat () {
    base_index = nullptr;
    own_fields = false;
    k_factor = 1;
}


void IndexRefineFlat::train (idx_t n, const float *x)
{
    base_index->train (n, x);
    is_trained = true;
}

void IndexRefineFlat::add (idx_t n, const float *x) {
    FAISS_THROW_IF_NOT (is_trained);
    base_index->add (n, x);
    refine_index.add (n, x);
    ntotal = refine_index.ntotal;
}

void IndexRefineFlat::reset ()
{
    base_index->reset ();
    refine_index.reset ();
    ntotal = 0;
}

namespace {
typedef faiss::Index::idx_t idx_t;

template<class C>
static void reorder_2_heaps (
      idx_t n,
      idx_t k, idx_t *labels, float *distances,
      idx_t k_base, const idx_t *base_labels, const float *base_distances)
{
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        idx_t *idxo = labels + i * k;
        float *diso = distances + i * k;
        const idx_t *idxi = base_labels + i * k_base;
        const float *disi = base_distances + i * k_base;

        heap_heapify<C> (k, diso, idxo, disi, idxi, k);
        if (k_base != k) { // add remaining elements
            heap_addn<C> (k, diso, idxo, disi + k, idxi + k, k_base - k);
        }
        heap_reorder<C> (k, diso, idxo);
    }
}


}


void IndexRefineFlat::search (
              idx_t n, const float *x, idx_t k,
              float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    idx_t k_base = idx_t (k * k_factor);
    idx_t * base_labels = labels;
    float * base_distances = distances;
    ScopeDeleter<idx_t> del1;
    ScopeDeleter<float> del2;


    if (k != k_base) {
        base_labels = new idx_t [n * k_base];
        del1.set (base_labels);
        base_distances = new float [n * k_base];
        del2.set (base_distances);
    }

    base_index->search (n, x, k_base, base_distances, base_labels);

    for (int i = 0; i < n * k_base; i++)
        assert (base_labels[i] >= -1 &&
                base_labels[i] < ntotal);

    // compute refined distances
    refine_index.compute_distance_subset (
        n, x, k_base, base_distances, base_labels);

    // sort and store result
    if (metric_type == METRIC_L2) {
        typedef CMax <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);

    } else if (metric_type == METRIC_INNER_PRODUCT) {
        typedef CMin <float, idx_t> C;
        reorder_2_heaps<C> (
            n, k, labels, distances,
            k_base, base_labels, base_distances);
    }

}



IndexRefineFlat::~IndexRefineFlat ()
{
    if (own_fields) delete base_index;
}

/***************************************************
 * IndexFlat1D
 ***************************************************/


IndexFlat1D::IndexFlat1D (bool continuous_update):
    IndexFlatL2 (1),
    continuous_update (continuous_update)
{
}

/// if not continuous_update, call this between the last add and
/// the first search
void IndexFlat1D::update_permutation ()
{
    perm.resize (ntotal);
    if (ntotal < 1000000) {
        fvec_argsort (ntotal, xb.data(), (size_t*)perm.data());
    } else {
        fvec_argsort_parallel (ntotal, xb.data(), (size_t*)perm.data());
    }
}

void IndexFlat1D::add (idx_t n, const float *x)
{
    IndexFlatL2::add (n, x);
    if (continuous_update)
        update_permutation();
}

void IndexFlat1D::reset()
{
    IndexFlatL2::reset();
    perm.clear();
}

void IndexFlat1D::search (
            idx_t n,
            const float *x,
            idx_t k,
            float *distances,
            idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG (perm.size() == ntotal,
                    "Call update_permutation before search");

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {

        float q = x[i]; // query
        float *D = distances + i * k;
        idx_t *I = labels + i * k;

        // binary search
        idx_t i0 = 0, i1 = ntotal;
        idx_t wp = 0;

        if (xb[perm[i0]] > q) {
            i1 = 0;
            goto finish_right;
        }

        if (xb[perm[i1 - 1]] <= q) {
            i0 = i1 - 1;
            goto finish_left;
        }

        while (i0 + 1 < i1) {
            idx_t imed = (i0 + i1) / 2;
            if (xb[perm[imed]] <= q) i0 = imed;
            else                    i1 = imed;
        }

        // query is between xb[perm[i0]] and xb[perm[i1]]
        // expand to nearest neighs

        while (wp < k) {
            float xleft = xb[perm[i0]];
            float xright = xb[perm[i1]];

            if (q - xleft < xright - q) {
                D[wp] = q - xleft;
                I[wp] = perm[i0];
                i0--; wp++;
                if (i0 < 0) { goto finish_right; }
            } else {
                D[wp] = xright - q;
                I[wp] = perm[i1];
                i1++; wp++;
                if (i1 >= ntotal) { goto finish_left; }
            }
        }
        goto done;

    finish_right:
        // grow to the right from i1
        while (wp < k) {
            if (i1 < ntotal) {
                D[wp] = xb[perm[i1]] - q;
                I[wp] = perm[i1];
                i1++;
            } else {
                D[wp] = 1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
        goto done;

    finish_left:
        // grow to the left from i0
        while (wp < k) {
            if (i0 >= 0) {
                D[wp] = q - xb[perm[i0]];
                I[wp] = perm[i0];
                i0--;
            } else {
                D[wp] = 1.0 / 0.0;
                I[wp] = -1;
            }
            wp++;
        }
    done:  ;
    }

}



} // namespace faiss
