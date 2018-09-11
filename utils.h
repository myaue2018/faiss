/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Copyright 2004-present Facebook. All Rights Reserved
 * -*- c++ -*-
 *
 *  A few utilitary functions for similarity search:
 * - random generators
 * - optimized exhaustive distance and knn search functions
 * - some functions reimplemented from torch for speed
 */

#ifndef FAISS_utils_h
#define FAISS_utils_h

#include <stdint.h>
// for the random data struct
#include <cstdlib>

#include "Heap.h"


namespace faiss {


/**************************************************
 * Get some stats about the system
**************************************************/


/// ms elapsed since some arbitrary epoch
double getmillisecs ();

/// get current RSS usage in kB
size_t get_mem_usage_kb ();


/**************************************************
 * Custom container includes a group of vectors
 **************************************************/

const size_t BLOCK_SIZE = 65536 * 384;

template <class T>
class GroupVector {
public:
    explicit GroupVector(size_t block_size = BLOCK_SIZE) : block_size(block_size) {}

    // append data to the tail of groups
    void append(const T* val, size_t n) {
        const T* p_val = val;
        size_t left_num = n;

        if (left_num == 0 || val == nullptr) {
            return;
        }
        if (data_.empty()) {
            data_.emplace_back(std::vector<T>());
            data_.reserve(block_size);
        }

        // fill the last block
        auto& tail_block = *(data_.rbegin());
        size_t available_size = block_size - tail_block.size();
        size_t tail_added_num = left_num > available_size ? available_size : left_num;
        tail_block.insert(tail_block.end(), p_val, p_val + tail_added_num);
        left_num -= tail_added_num;
        p_val += tail_added_num;

        if (left_num > 0) {
            // new block and fill the entire block
            size_t block_added_num = left_num / block_size;
            for (size_t i = 0; i < block_added_num; ++i) {
                data_.emplace_back(std::vector<T>());
                auto& curr_block = *(data_.rbegin());
                curr_block.reserve(block_size);
                curr_block.insert(curr_block.end(), p_val, p_val + block_size);
                left_num -= block_size;
                p_val += block_size;
            }

            // store the left data into the last new block
            if (left_num > 0) {
                data_.emplace_back(std::vector<T>());
                auto& last_block = *(data_.rbegin());
                last_block.reserve(block_size);
                last_block.insert(last_block.end(), p_val, p_val + left_num);
            }
        }
    }

    // replace data by the ordered index idx
    void replace(size_t idx, const T* val, size_t num) const {
        auto& dst_block = data_[idx / block_size];
        memcpy((T*)dst_block.data() + idx % block_size, val, num);
    }

    // adjust the group size to fit n and reserve the last block to block_size
    void resize(size_t n) {
        size_t pos = n / block_size;
        if (n % block_size == 0) {
            data_.resize(pos);
        } else {
            data_.resize(pos + 1);
            data_.rbegin()->resize(n - pos * block_size);
            data_.rbegin()->reserve(block_size);
        }
    }

    // adjust the group capacity if necessary and reserve the last block to block_size
    void reserve(size_t n) {
        size_t new_size = (n % block_size == 0) ? n / block_size : n / block_size + 1;
        data_.reserve(new_size);
    }
    void clear() { data_.clear(); }

    // return the data at the ordered index idx
    inline T& operator[](size_t idx) const { return data_[idx / block_size][idx % block_size]; }
    inline T& operator[](size_t idx) { return data_[idx / block_size][idx % block_size]; }
    // return the total num of data stored in the group vector
    inline size_t size() const { return (data_.size() - 1) * block_size + data_.rbegin().size(); }

    // functions used to block level access
    inline const std::vector<T>& blockAt(size_t idx) const { return data_[idx]; }
    inline size_t blockNum() const { return data_.size(); }
    inline size_t blockSize() const { return block_size; }

private:
    std::vector<std::vector<T>> data_;
    size_t block_size = 0;
};


/**************************************************
 * Random data generation functions
 **************************************************/

/// random generator that can be used in multithreaded contexts
struct RandomGenerator {

#ifdef __linux__
    char rand_state [8];
    struct random_data rand_data;
#elif __APPLE__
    unsigned rand_state;
#endif

    /// random 31-bit positive integer
    int rand_int ();

    /// random long < 2 ^ 62
    long rand_long ();

    /// generate random number between 0 and max-1
    int rand_int (int max);

    /// between 0 and 1
    float rand_float ();


    double rand_double ();

    /// initialize
    explicit RandomGenerator (long seed = 1234);

    /// default copy constructor messes up pointer in rand_data
    RandomGenerator (const RandomGenerator & other);

};

/* Generate an array of uniform random floats / multi-threaded implementation */
void float_rand (float * x, size_t n, long seed);
void float_randn (float * x, size_t n, long seed);
void long_rand (long * x, size_t n, long seed);
void byte_rand (uint8_t * x, size_t n, long seed);

/* random permutation */
void rand_perm (int * perm, size_t n, long seed);



 /*********************************************************
 * Optimized distance/norm/inner prod computations
 *********************************************************/


/// Squared L2 distance between two vectors
float fvec_L2sqr (
        const float * x,
        const float * y,
        size_t d);

/* SSE-implementation of inner product and L2 distance */
float  fvec_inner_product (
        const float * x,
        const float * y,
        size_t d);


/// a balanced assignment has a IF of 1
double imbalance_factor (int n, int k, const long *assign);

/// same, takes a histogram as input
double imbalance_factor (int k, const int *hist);

/** Compute pairwise distances between sets of vectors
 *
 * @param d     dimension of the vectors
 * @param nq    nb of query vectors
 * @param nb    nb of database vectors
 * @param xq    query vectors (size nq * d)
 * @param xb    database vectros (size nb * d)
 * @param dis   output distances (size nq * nb)
 * @param ldq,ldb, ldd strides for the matrices
 */
void pairwise_L2sqr (long d,
                     long nq, const float *xq,
                     long nb, const float *xb,
                     float *dis,
                     long ldq = -1, long ldb = -1, long ldd = -1);


/* compute the inner product between nx vectors x and one y */
void fvec_inner_products_ny (
        float * ip,         /* output inner product */
        const float * x,
        const float * y,
        size_t d, size_t ny);

/* compute ny square L2 distance bewteen x and a set of contiguous y vectors */
void fvec_L2sqr_ny (
        float * __restrict dis,
        const float * x,
        const float * y,
        size_t d, size_t ny);


/** squared norm of a vector */
float fvec_norm_L2sqr (const float * x,
                       size_t d);

/** compute the L2 norms for a set of vectors
 *
 * @param  ip       output norms, size nx
 * @param  x        set of vectors, size nx * d
 */
void fvec_norms_L2 (float * ip, const float * x, size_t d, size_t nx);

/// same as fvec_norms_L2, but computes square norms
void fvec_norms_L2sqr (float * ip, const float * x, size_t d, size_t nx);

/* L2-renormalize a set of vector. Nothing done if the vector is 0-normed */
void fvec_renorm_L2 (size_t d, size_t nx, float * x);


/* This function exists because the Torch counterpart is extremly slow
   (not multi-threaded + unexpected overhead even in single thread).
   It is here to implement the usual property |x-y|^2=|x|^2+|y|^2-2<x|y>  */
void inner_product_to_L2sqr (float * __restrict dis,
                             const float * nr1,
                             const float * nr2,
                             size_t n1, size_t n2);

float fvec_norm_L2r_ref_int8 (const int8_t * x, size_t d);

void fvec_norms_L2r_ref_int8 (float * ip, const int8_t * x, size_t d, size_t nx);

float fvec_norm_L2r_ref_uint8 (const uint8_t * x, size_t d);

/***************************************************************************
 * Compute a subset of  distances
 ***************************************************************************/

 /* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_inner_products_by_idx (
        float * __restrict ip,
        const float * x,
        const float * y,
        const long * __restrict ids,
        size_t d, size_t nx, size_t ny);

/* same but for a subset in y indexed by idsy (ny vectors in total) */
void fvec_L2sqr_by_idx (
        float * __restrict dis,
        const float * x,
        const float * y,
        const long * __restrict ids, /* ids of y vecs */
        size_t d, size_t nx, size_t ny);

/***************************************************************************
 * KNN functions
 ***************************************************************************/

// threshold on nx above which we switch to BLAS to compute distances
extern int distance_compute_blas_threshold;

// convert float to unsigned int8
void FloatToUint8 (uint8_t* out,
                   const float* in,
                   size_t num);

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result array, which also provides k. Sorted on output
 */
void knn_inner_product (
        const float * x,
        const float * y,
        size_t d, size_t nx, size_t ny,
        float_minheap_array_t * res);

void knn_inner_product (const float * x,
                        const GroupVector<uint8_t> & yb,
                        size_t d, size_t nx,
                        float_minheap_array_t * res,
                        float* queryNorms_,
                        bool ignore_negative);

/** Same as knn_inner_product, for the L2 distance */
void knn_L2sqr (
        const float * x,
        const float * y,
        size_t d, size_t nx, size_t ny,
        float_maxheap_array_t * res);

/** same as knn_L2sqr, but base_shift[bno] is subtracted to all
 * computed distances.
 *
 * @param base_shift   size ny
 */
void knn_L2sqr_base_shift (
         const float * x,
         const float * y,
         size_t d, size_t nx, size_t ny,
         float_maxheap_array_t * res,
         const float *base_shift);

/* Find the nearest neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 */
void knn_inner_products_by_idx (
        const float * x,
        const float * y,
        const long *  ids,
        size_t d, size_t nx, size_t ny,
        float_minheap_array_t * res);

void knn_L2sqr_by_idx (const float * x,
                       const float * y,
                       const long * __restrict ids,
                       size_t d, size_t nx, size_t ny,
                       float_maxheap_array_t * res);

/***************************************************************************
 * Range search
 ***************************************************************************/



/// Forward declaration, see AuxIndexStructures.h
struct RangeSearchResult;

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x      query vectors, size nx * d
 * @param y      database vectors, size ny * d
 * @param radius search radius around the x vectors
 * @param result result structure
 */
void range_search_L2sqr (
        const float * x,
        const float * y,
        size_t d, size_t nx, size_t ny,
        float radius,
        RangeSearchResult *result);

/// same as range_search_L2sqr for the inner product similarity
void range_search_inner_product (
        const float * x,
        const float * y,
        size_t d, size_t nx, size_t ny,
        float radius,
        RangeSearchResult *result);





/***************************************************************************
 * Misc  matrix and vector manipulation functions
 ***************************************************************************/


/** compute c := a + bf * b for a, b and c tables
 *
 * @param n   size of the tables
 * @param a   size n
 * @param b   size n
 * @param c   restult table, size n
 */
void fvec_madd (size_t n, const float *a,
                float bf, const float *b, float *c);


/** same as fvec_madd, also return index of the min of the result table
 * @return    index of the min of table c
 */
int fvec_madd_and_argmin (size_t n, const float *a,
                           float bf, const float *b, float *c);


/* perform a reflection (not an efficient implementation, just for test ) */
void reflection (const float * u, float * x, size_t n, size_t d, size_t nu);


/** For k-means: update stage.
 *
 * @param x          training vectors, size n * d
 * @param centroids  centroid vectors, size k * d
 * @param assign     nearest centroid for each training vector, size n
 * @param k_frozen   do not update the k_frozen first centroids
 * @return           nb of spliting operations to fight empty clusters
 */
int km_update_centroids (
        const float * x,
        float * centroids,
        long * assign,
        size_t d, size_t k, size_t n,
        size_t k_frozen);

/** compute the Q of the QR decomposition for m > n
 * @param a   size n * m: input matrix and output Q
 */
void matrix_qr (int m, int n, float *a);

/** distances are supposed to be sorted. Sorts indices with same distance*/
void ranklist_handle_ties (int k, long *idx, const float *dis);

/** count the number of comon elements between v1 and v2
 * algorithm = sorting + bissection to avoid double-counting duplicates
 */
size_t ranklist_intersection_size (size_t k1, const long *v1,
                                   size_t k2, const long *v2);

/** merge a result table into another one
 *
 * @param I0, D0       first result table, size (n, k)
 * @param I1, D1       second result table, size (n, k)
 * @param keep_min     if true, keep min values, otherwise keep max
 * @param translation  add this value to all I1's indexes
 * @return             nb of values that were taken from the second table
 */
size_t merge_result_table_with (size_t n, size_t k,
                                long *I0, float *D0,
                                const long *I1, const float *D1,
                                bool keep_min = true,
                                long translation = 0);



void fvec_argsort (size_t n, const float *vals,
                    size_t *perm);

void fvec_argsort_parallel (size_t n, const float *vals,
                    size_t *perm);


/// compute histogram on v
int ivec_hist (size_t n, const int * v, int vmax, int *hist);

/** Compute histogram of bits on a code array
 *
 * @param codes   size(n, nbits / 8)
 * @param hist    size(nbits): nb of 1s in the array of codes
 */
void bincode_hist(size_t n, size_t nbits, const uint8_t *codes, int *hist);


/// compute a checksum on a table.
size_t ivec_checksum (size_t n, const int *a);


/** random subsamples a set of vectors if there are too many of them
 *
 * @param d      dimension of the vectors
 * @param n      on input: nb of input vectors, output: nb of output vectors
 * @param nmax   max nb of vectors to keep
 * @param x      input array, size *n-by-d
 * @param seed   random seed to use for sampling
 * @return       x or an array allocated with new [] with *n vectors
 */
const float *fvecs_maybe_subsample (
       size_t d, size_t *n, size_t nmax, const float *x,
       bool verbose = false, long seed = 1234);

} // namspace faiss


#endif /* FAISS_utils_h */
