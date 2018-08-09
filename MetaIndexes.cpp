/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "MetaIndexes.h"

#include <pthread.h>

#include <cstdio>
#include <algorithm>
#include <glog/logging.h>

#include "FaissAssert.h"
#include "Heap.h"
#include "AuxIndexStructures.h"


namespace faiss {

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

IndexIDMap::IndexIDMap (Index *index):
    index (index),
    own_fields (false)
{
    FAISS_THROW_IF_NOT_MSG (index->ntotal == 0, "index must be empty on input");
    is_trained = index->is_trained;
    metric_type = index->metric_type;
    verbose = index->verbose;
    d = index->d;
}

void IndexIDMap::add (idx_t, const float *)
{
    FAISS_THROW_MSG ("add does not make sense with IndexIDMap, "
                      "use add_with_ids");
}


void IndexIDMap::train (idx_t n, const float *x)
{
    index->train (n, x);
    is_trained = index->is_trained;
}

void IndexIDMap::reset ()
{
    index->reset ();
    ntotal = 0;
}


void IndexIDMap::add_with_ids (idx_t n, const float * x, const long *xids)
{
    index->add (n, x);
    error_state = index->get_error_state();
    if (error_state != Faiss_Error_OK)
        return;
    for (idx_t i = 0; i < n; i++)
        id_map.push_back (xids[i]);
    ntotal = index->ntotal;
}


void IndexIDMap::search (idx_t n, const float *x, idx_t k,
                              float *distances, idx_t *labels) const
{
    index->search (n, x, k, distances, labels);
    idx_t *li = labels;
    for (idx_t i = 0; i < n * k; i++) {
        li[i] = li[i] < 0 ? li[i] : id_map[li[i]];
    }
}


void IndexIDMap::range_search (idx_t n, const float *x, float radius,
                   RangeSearchResult *result) const
{
  index->range_search(n, x, radius, result);
  for (idx_t i = 0; i < result->lims[result->nq]; i++) {
      result->labels[i] = result->labels[i] < 0 ?
        result->labels[i] : id_map[result->labels[i]];
  }
}

namespace {

struct IDTranslatedSelector: IDSelector {
    const std::vector <long> & id_map;
    const IDSelector & sel;
    IDTranslatedSelector (const std::vector <long> & id_map,
                          const IDSelector & sel):
        id_map (id_map), sel (sel)
    {}
    bool is_member(idx_t id) const override {
      return sel.is_member(id_map[id]);
    }
};

}

long IndexIDMap::remove_ids (const IDSelector & sel)
{
    // remove in sub-index first
    IDTranslatedSelector sel2 (id_map, sel);
    long nremove = index->remove_ids (sel2);

    long j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member (id_map[i])) {
            // remove
        } else {
            id_map[j] = id_map[i];
            j++;
        }
    }
    FAISS_ASSERT (j == index->ntotal);
    ntotal = j;
    id_map.resize(ntotal);
    return nremove;
}

void IndexIDMap::set_user_reserve(bool is_reserve)
{
    Index::set_user_reserve(is_reserve);
    index->set_user_reserve(is_reserve);
}

void IndexIDMap::set_use_int8_norms(bool flag)
{
    index->set_use_int8_norms(flag);
}

void IndexIDMap::set_index_int8_cosine_ignore_negative(bool flag)
{
    index->set_index_int8_cosine_ignore_negative(flag);
}

IndexIDMap::~IndexIDMap ()
{
    if (own_fields) delete index;
}

void IndexIDMap::set_max_size(size_t new_size) {
    Index::set_max_size(new_size);
    index->set_max_size(new_size);
}

/*****************************************************
 * IndexIDMap2 implementation
 *******************************************************/

IndexIDMap2::IndexIDMap2 (Index *index): IndexIDMap (index)
{}

void IndexIDMap2::add_with_ids(idx_t n, const float* x, const long* xids)
{
    size_t prev_ntotal = ntotal;
    IndexIDMap::add_with_ids (n, x, xids);
    if (error_state != Faiss_Error_OK)
        return;
    for (size_t i = prev_ntotal; i < ntotal; i++) {
        rev_map [id_map [i]] = i;
    }
}

void IndexIDMap2::construct_rev_map ()
{
    rev_map.clear ();
    for (size_t i = 0; i < ntotal; i++) {
        rev_map [id_map [i]] = i;
    }
}
int  IndexIDMap2::reserve(faiss::Index::idx_t n){
    auto a =  index->reserve(n);
    error_state = index->get_error_state();
    return a;
}
long IndexIDMap2::remove_ids(const idx_t & idx)
{
    if(rev_map.find(idx)==rev_map.end()){
        return -1;
    }
    auto id_real = rev_map.at(idx);

    auto old_size = id_map.size();
    if(old_size==1){
        id_map.clear();
        rev_map.clear();
    }else{
        auto last_value = id_map[old_size-1];
        id_map[rev_map.at(idx)] = last_value;
        id_map.resize(old_size-1);

        rev_map[last_value] = rev_map.at(idx);
        rev_map.erase(idx);
    }
    auto ret = index->remove_ids(id_real);
    error_state = index->get_error_state();
    ntotal = index->ntotal;
    return ret;
//    // This is quite inefficient
//    long nremove = IndexIDMap::remove_ids (sel);
//    construct_rev_map ();
//    return nremove;
}

long IndexIDMap2::remove_ids(const IDSelector& sel)
{

    //只支持删除一个
    if(sel.single_id==-1)
        FAISS_THROW_MSG ("remove_ids not implemented for non-single");

    if(rev_map.find(sel.single_id)==rev_map.end()){
        return -1;
    }

    auto old_size = id_map.size();
    if(old_size==1){
        id_map.clear();
        rev_map.clear();
    }else{
        auto last_value = id_map[old_size-1];
        id_map[rev_map.at(sel.single_id)] = last_value;
        id_map.resize(old_size-1);

        rev_map[last_value] = rev_map.at(sel.single_id);
        rev_map.erase(sel.single_id);
    }

    return index->remove_ids(sel);
//    // This is quite inefficient
//    long nremove = IndexIDMap::remove_ids (sel);
//    construct_rev_map ();
//    return nremove;
}

void IndexIDMap2::reconstruct (idx_t key, float * recons) const
{
    try {
        index->reconstruct (rev_map.at (key), recons);
    } catch (const std::out_of_range& e) {
        FAISS_THROW_FMT ("key %ld not found", key);
    }
}

void IndexIDMap2::update(idx_t key,const float * new_f) const
{
    if(rev_map.find(key)==rev_map.end()){
        return;
    }
    index->update((rev_map.at(key)),new_f);
    ((IndexIDMap2*)(this))->error_state = index->error_state;
}

void IndexIDMap2::get_query_norms(float *query_norms)
{
    index->get_query_norms(query_norms);
}

void IndexIDMap2::get_feature_norms(idx_t n, idx_t k, const idx_t *ids, float *feature_norms)
{
    std::vector<idx_t> rev_ids(static_cast<unsigned long>(n * k));
    for (idx_t i = 0; i < n * k; ++i) {
        rev_ids[i] = rev_map[ids[i]];
    }
    index->get_feature_norms(n, k, rev_ids.data(), feature_norms);
}



/*****************************************************
 * IndexShards implementation
 *******************************************************/

// subroutines
namespace {


typedef Index::idx_t idx_t;


template<class Job>
struct Thread {
    Job job;
    pthread_t thread;
    bool error = false;

    Thread () {}

    explicit Thread (const Job & job): job(job) {}

    bool start () {
        auto ret = pthread_create (&thread, nullptr, run, this);
        if(ret!=0){
            PLOG(ERROR) << "[pthread_create]error:"<<ret;
            return false;
        }
        return true;
    }

    void wait () {
        pthread_join (thread, nullptr);
    }

    static void * run (void *arg) {
        static_cast<Thread*> (arg)->job.run();
        return nullptr;
    }

};


/// callback + thread management to train 1 shard
struct TrainJob {
    IndexShards *index;    // the relevant index
    int no;                // shard number
    idx_t n;               // train points
    const float *x;

    void run ()
    {
        if (index->verbose)
            printf ("begin train shard %d on %ld points\n", no, n);
        index->shard_indexes [no]->train(n, x);
        if (index->verbose)
            printf ("end train shard %d\n", no);
    }

};

struct AddJob {
    IndexShards *index;    // the relevant index
    int no;                // shard number

    idx_t n;
    const float *x;
    const idx_t *ids;

    void run ()
    {
        if (index->verbose)
            printf ("begin add shard %d on %ld points\n", no, n);
        if (ids)
            index->shard_indexes[no]->add_with_ids (n, x, ids);
        else
            index->shard_indexes[no]->add (n, x);

        if(index->shard_indexes[no]->error_state!=Faiss_Error_OK){
            index->error_state = index->shard_indexes[no]->get_error_state();
        }
        if (index->verbose)
            printf ("end add shard %d on %ld points\n", no, n);
    }
};



/// callback + thread management to query in 1 shard
struct QueryJob {
    IndexShards *index;    // the relevant index
    int no;                // shard number

    // query params
    idx_t n;
    const float *x;
    idx_t k;
    float *distances;
    idx_t *labels;

    MetricType metric_type;


    void run ()
    {
        if (index->verbose)
            printf ("begin query shard %d on %ld points\n", no, n);


        idx_t k_real = k;

        if(k_real > index->shard_indexes [no]->ntotal)
        {
            k_real = index->shard_indexes [no]->ntotal;
        }

        index->shard_indexes [no]->search (n, x, k_real,
                                           distances, labels);

        if(k_real<k)
        {
            for (int i = k_real; i < k; ++i) {
                if (metric_type == METRIC_L2) {
                    distances[i] = std::numeric_limits<float>::max();
                    labels[i] = 0;
                } else {
                    distances[i] = std::numeric_limits<float>::min();
                    labels[i] = 0;
                }
            }
        }

        if(index->shard_indexes[no]->error_state!=Faiss_Error_OK){
            index->error_state = index->shard_indexes[no]->get_error_state();
        }
        if (index->verbose)
            printf ("end query shard %d\n", no);
    }


};




// add translation to all valid labels
void translate_labels (long n, idx_t *labels, long translation)
{
    if (translation == 0) return;
    for (long i = 0; i < n; i++) {
        if(labels[i] < 0) return;
        labels[i] += translation;
    }
}


/** merge result tables from several shards.
 * @param all_distances  size nshard * n * k
 * @param all_labels     idem
 * @param translartions  label translations to apply, size nshard
 */

template <class C>
void merge_tables (long n, long k, long nshard,
                   float *distances, idx_t *labels,
                   const float *all_distances,
                   idx_t *all_labels,
                   const long *translations)
{
    if(k == 0) {
        return;
    }

    long stride = n * k;
#pragma omp parallel
    {
        std::vector<int> buf (2 * nshard);
        int * pointer = buf.data();
        int * shard_ids = pointer + nshard;
        std::vector<float> buf2 (nshard);
        float * heap_vals = buf2.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the shard where they are
            // produced.
            const float *D_in = all_distances + i * k;
            const idx_t *I_in = all_labels + i * k;
            int heap_size = 0;

            for (long s = 0; s < nshard; s++) {
                pointer[s] = 0;
                if (I_in[stride * s] >= 0)
                    heap_push<C> (++heap_size, heap_vals, shard_ids,
                                 D_in[stride * s], s);
            }

            float *D = distances + i * k;
            idx_t *I = labels + i * k;

            for (int j = 0; j < k; j++) {
                if (heap_size == 0) {
                    I[j] = -1;
                    D[j] = C::neutral();
                } else {
                    // pop best element
                    int s = shard_ids[0];
                    int & p = pointer[s];
                    D[j] = heap_vals[0];
                    I[j] = I_in[stride * s + p] + translations[s];

                    heap_pop<C> (heap_size--, heap_vals, shard_ids);
                    p++;
                    if (p < k && I_in[stride * s + p] >= 0)
                        heap_push<C> (++heap_size, heap_vals, shard_ids,
                                     D_in[stride * s + p], s);
                }
            }
        }
    }
}

    /** merge result tables from several shards.
 * @param all_distances  size nshard * n * k
 * @param all_labels     idem
 * @param translartions  label translations to apply, size nshard
 */

    template <class C>
    void merge_tables (long n, long k, long nshard,
                       float *distances, idx_t *labels,
                       const float *all_distances,
                       idx_t *all_labels,
                       const long *translations,int * pointer,float * heap_vals)
    {
        if(k == 0) {
            return;
        }

        long stride = n * k;
//#pragma omp parallel
        {
//            std::vector<int> buf (2 * nshard);
//            int * pointer = buf.data();
            int * shard_ids = pointer + nshard;
//            std::vector<float> buf2 (nshard);
//            float * heap_vals = buf2.data();
//#pragma omp for
            for (long i = 0; i < n; i++) {
                // the heap maps values to the shard where they are
                // produced.
                const float *D_in = all_distances + i * k;
                const idx_t *I_in = all_labels + i * k;
                int heap_size = 0;

                for (long s = 0; s < nshard; s++) {
                    pointer[s] = 0;
                    if (I_in[stride * s] >= 0)
                        heap_push<C> (++heap_size, heap_vals, shard_ids,
                                      D_in[stride * s], s);
                }

                float *D = distances + i * k;
                idx_t *I = labels + i * k;

                for (int j = 0; j < k; j++) {
                    if (heap_size == 0) {
                        I[j] = -1;
                        D[j] = C::neutral();
                    } else {
                        // pop best element
                        int s = shard_ids[0];
                        int & p = pointer[s];
                        D[j] = heap_vals[0];
                        I[j] = I_in[stride * s + p] + translations[s];

                        heap_pop<C> (heap_size--, heap_vals, shard_ids);
                        p++;
                        if (p < k && I_in[stride * s + p] >= 0)
                            heap_push<C> (++heap_size, heap_vals, shard_ids,
                                          D_in[stride * s + p], s);
                    }
                }
            }
        }
    }


};






IndexShards::IndexShards (idx_t d, bool threaded, bool successive_ids):
    Index (d), own_fields (false),lastAddedIndex(0),
    threaded (threaded), successive_ids (successive_ids)
{
    heap_temp_int.resize(20*2);
    heap_temp_float.resize(20*2);
}

void IndexShards::add_shard (Index *idx)
{
    shard_indexes.push_back (idx);
    all_distances_v.resize(MAX_TOPK*MAX_N_QUERY*shard_indexes.size());
    all_labels_v.resize(MAX_TOPK*MAX_N_QUERY*shard_indexes.size());

    sync_with_shard_indexes ();
}

void IndexShards::sync_with_shard_indexes ()
{
    if (shard_indexes.empty()) return;
    Index * index0 = shard_indexes[0];
    d = index0->d;
    metric_type = index0->metric_type;
    is_trained = index0->is_trained;
    ntotal = index0->ntotal;
    for (int i = 1; i < shard_indexes.size(); i++) {
        Index * index = shard_indexes[i];
        FAISS_THROW_IF_NOT (metric_type == index->metric_type);
        FAISS_THROW_IF_NOT (d == index->d);
        ntotal += index->ntotal;
    }
}


void IndexShards::train (idx_t n, const float *x)
{

    // pre-alloc because we don't want reallocs
    std::vector<Thread<TrainJob > > tss (shard_indexes.size());
    int nt = 0;
    for (int i = 0; i < shard_indexes.size(); i++) {
        if(!shard_indexes[i]->is_trained) {
            TrainJob ts = {this, i, n, x};
            if (threaded) {
                tss[nt] = Thread<TrainJob> (ts);
                tss[nt++].start();
            } else {
                ts.run();
            }
        }
    }
    for (int i = 0; i < nt; i++) {
        tss[i].wait();
    }
    sync_with_shard_indexes ();
}

void IndexShards::add (idx_t n, const float *x)
{
    add_with_ids (n, x, nullptr);
}



typedef  struct{
    int index=0;
    int64_t size=0;
    int64_t add_size=0;
}ShardSize;

bool comp(const ShardSize &a,const ShardSize &b)
{
    return a.size<b.size;
}

 /**
  * Cases (successive_ids, xids):
  * - true, non-NULL       ERROR: it makes no sense to pass in ids and
  *                        request them to be shifted
  * - true, NULL           OK, but should be called only once (calls add()
  *                        on sub-indexes).
  * - false, non-NULL      OK: will call add_with_ids with passed in xids
  *                        distributed evenly over shards
  * - false, NULL          OK: will call add_with_ids on each sub-index,
  *                        starting at ntotal
  */

void IndexShards::add_with_ids (idx_t n, const float * x, const long *xids)
{

    FAISS_THROW_IF_NOT_MSG(!(successive_ids && xids),
                   "It makes no sense to pass in ids and "
                   "request them to be shifted");

    if (successive_ids) {
      FAISS_THROW_IF_NOT_MSG(!xids,
                       "It makes no sense to pass in ids and "
                       "request them to be shifted");
      FAISS_THROW_IF_NOT_MSG(ntotal == 0,
                       "when adding to IndexShards with sucessive_ids, "
                       "only add() in a single pass is supported");
    }

    long nshard = shard_indexes.size();
    const long *ids = xids;
    ScopeDeleter<long> del;
    if (!ids && !successive_ids) {
        long *aids = new long[n];
        for (long i = 0; i < n; i++)
            aids[i] = ntotal + i;
        ids = aids;
        del.set (ids);
    }

    std::vector<Thread<AddJob > > asa (shard_indexes.size());
    std::vector<bool> isNeedStart (shard_indexes.size());

    std::vector<ShardSize> nshard_info(nshard+1);
    {//计算每个shard要添加多少，尽可能平均
        int left_feature_size = n;
        for (int i = 0; i < nshard; i++) {
            ShardSize& shardSize = nshard_info[i];
            shardSize.index = i;
            shardSize.size = shard_indexes[i]->ntotal;
            shardSize.add_size = 0;
        }
        nshard_info[nshard].index =-1;
        nshard_info[nshard].size = MAX_ADD_BATCH;
        nshard_info[nshard].add_size =0;


        std::sort(nshard_info.begin(),nshard_info.end(),comp);

        for (int i = 0; i < nshard; i++) {
            auto dt = nshard_info[i+1].size - nshard_info[i].size;
            if(dt<=0){
                continue;
            }
            if(dt*(i+1) <= left_feature_size){
                for (int j = 0; j <= i ; ++j) {
                    nshard_info[j].add_size +=dt;
                }
                left_feature_size-=dt*(i+1);
                continue;
            }else{
                nshard_info[0].add_size += left_feature_size%(i+1);
                for (int j = 0; j <= i ; ++j) {
                    nshard_info[j].add_size +=left_feature_size/(i+1);
                }
                left_feature_size = 0;
                break;
            }
        }
    }


    //fid2sid_map
    int start_size = 0;

    for (int i = 0; i < nshard; i++) {
        int sub_index_id = nshard_info[i].index;
        long i0 = start_size  ;
        long i1 = start_size+nshard_info[i].add_size;
        start_size += nshard_info[i].add_size;

        if(i1 - i0 >0){
            isNeedStart[sub_index_id] = true;
        }else{
            isNeedStart[sub_index_id] = false;
            continue;
        }

        for (int j = 0; j < i1 - i0; ++j) {
            fid2sid_map[(idx_t)*(ids + i0+j)]=sub_index_id;
        }

        AddJob as = {this, sub_index_id,
                       i1 - i0, x + i0 * d,
                       ids ? ids + i0 : nullptr};
        if (threaded) {
            asa[sub_index_id] = Thread<AddJob>(as);
            asa[sub_index_id].start();
        } else {
            as.run();
        }
    }
    for (int i = 0; i < nshard; i++) {
        if(isNeedStart[i] && threaded){
            asa[i].wait();
        }
    }
    ntotal += n;
}





void IndexShards::reset ()
{
    for (int i = 0; i < shard_indexes.size(); i++) {
        shard_indexes[i]->reset ();
    }
    fid2sid_map.clear();
    sync_with_shard_indexes ();
}

void IndexShards::search (
           idx_t n, const float *x, idx_t k,
           float *distances, idx_t *labels) const
{
    long nshard = shard_indexes.size();
//    float *all_distances = new float [nshard * k * n];
//    idx_t *all_labels = new idx_t [nshard * k * n];
    float *all_distances = (float *)all_distances_v.data();
    idx_t *all_labels = (idx_t*)all_labels_v.data();
//    ScopeDeleter<float> del (all_distances);
//    ScopeDeleter<idx_t> del2 (all_labels);

#if 1

    // pre-alloc because we don't want reallocs
    std::vector<Thread<QueryJob> > qss (nshard);
    for (int i = 0; i < nshard; i++) {
        QueryJob qs = {
                (IndexShards*)this, i, n, x, k,
            all_distances + i * k * n,
            all_labels + i * k * n,
            metric_type
        };
        if (threaded) {
            qss[i] = Thread<QueryJob> (qs);
            auto ret = qss[i].start();
            if(!ret){
                qs.run();
                qss[i].error = true;
            }
        } else {
            qs.run();
        }
    }

    if (threaded) {
        for (int i = 0; i < qss.size(); i++) {
            if(qss[i].error==false){
                qss[i].wait();
            }
        }
    }
#else

    // pre-alloc because we don't want reallocs
    std::vector<QueryJob> qss (nshard);
    for (int i = 0; i < nshard; i++) {
        QueryJob qs = {
            this, i, n, x, k,
            all_distances + i * k * n,
            all_labels + i * k * n
        };
        if (threaded) {
            qss[i] = qs;
        } else {
            qs.run();
        }
    }

    if (threaded) {
#pragma omp parallel for
        for (int i = 0; i < qss.size(); i++) {
            qss[i].run();
        }
    }

#endif
    std::vector<long> translations (nshard, 0);
    if (successive_ids) {
        translations[0] = 0;
        for (int s = 0; s + 1 < nshard; s++)
            translations [s + 1] = translations [s] +
                shard_indexes [s]->ntotal;
    }

    if (metric_type == METRIC_L2) {
        merge_tables< CMin<float, int> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data ());
    } else {
        merge_tables< CMax<float, int> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data (),
             (int*)(heap_temp_int.data()),(float*)(heap_temp_float.data()));
    }

}



IndexShards::~IndexShards ()
{
    if (own_fields) {
        for (int s = 0; s < shard_indexes.size(); s++)
            delete shard_indexes [s];
    }
}

long IndexShards::remove_ids(const idx_t &idx) {
    auto iter = fid2sid_map.find(idx);
    if(iter==fid2sid_map.end()){
        return -1;
    }
    auto sub_index = shard_indexes[iter->second];
    fid2sid_map.erase(iter);

    auto n_src = sub_index->ntotal;
    auto rt =  sub_index->remove_ids(idx);
    ntotal -= n_src - rt;
    error_state = sub_index->get_error_state();
    return rt;
}

int IndexShards::reserve(faiss::Index::idx_t n) {
    for (int s = 0; s < shard_indexes.size(); s++)
    {
        shard_indexes [s]->reserve(n/(shard_indexes.size())+1);
        if(shard_indexes [s]->error_state!=Faiss_Error_OK){
            error_state = shard_indexes [s]->get_error_state();
        }
    }

    return 0;
}

void IndexShards::update(idx_t key, const float *recons) const {
    auto iter = fid2sid_map.find(key);
    if(iter==fid2sid_map.end()){
        return ;
    }
    auto sub_index = shard_indexes[iter->second];
    sub_index->update(key,recons);
    ((IndexShards*)(this))->error_state = sub_index->get_error_state();
}

void IndexShards::set_user_reserve(bool is_reserve) {
    Index::set_user_reserve(is_reserve);
    for (int i = 0; i < shard_indexes.size(); ++i) {
        shard_indexes[i]->set_user_reserve(is_reserve);
    }
}

void IndexShards::set_max_size(size_t new_size) {
    Index::set_max_size(new_size);
    for (int i = 0; i < shard_indexes.size(); ++i) {
        shard_indexes[i]->set_max_size(new_size/(shard_indexes.size())+1);
    }
}


/*****************************************************
 * IndexSplitVectors implementation
 *******************************************************/


IndexSplitVectors::IndexSplitVectors (idx_t d, bool threaded):
    Index (d), own_fields (false),
    threaded (threaded), sum_d (0)
{

}

void IndexSplitVectors::add_sub_index (Index *index)
{
    sub_indexes.push_back (index);
    sync_with_sub_indexes ();
}

void IndexSplitVectors::sync_with_sub_indexes ()
{
    if (sub_indexes.empty()) return;
    Index * index0 = sub_indexes[0];
    sum_d = index0->d;
    metric_type = index0->metric_type;
    is_trained = index0->is_trained;
    ntotal = index0->ntotal;
    for (int i = 1; i < sub_indexes.size(); i++) {
        Index * index = sub_indexes[i];
        FAISS_THROW_IF_NOT (metric_type == index->metric_type);
        FAISS_THROW_IF_NOT (ntotal == index->ntotal);
        sum_d += index->d;
    }

}

void IndexSplitVectors::add(idx_t /*n*/, const float* /*x*/) {
  FAISS_THROW_MSG("not implemented");
}

namespace {

/// callback + thread management to query in 1 shard
struct SplitQueryJob {
    const IndexSplitVectors *index;    // the relevant index
    int no;                // shard number

    // query params
    idx_t n;
    const float *x;
    idx_t k;
    float *distances;
    idx_t *labels;


    void run ()
    {
        if (index->verbose)
            printf ("begin query shard %d on %ld points\n", no, n);
        const Index * sub_index = index->sub_indexes[no];
        long sub_d = sub_index->d, d = index->d;
        idx_t ofs = 0;
        for (int i = 0; i < no; i++) ofs += index->sub_indexes[i]->d;
        float *sub_x = new float [sub_d * n];
        ScopeDeleter<float> del (sub_x);
        for (idx_t i = 0; i < n; i++)
            memcpy (sub_x + i * sub_d, x + ofs + i * d, sub_d * sizeof (sub_x));
        sub_index->search (n, sub_x, k, distances, labels);
        if (index->verbose)
            printf ("end query shard %d\n", no);
    }

};



}



void IndexSplitVectors::search (
           idx_t n, const float *x, idx_t k,
           float *distances, idx_t *labels) const
{
    FAISS_THROW_IF_NOT_MSG (k == 1,
                      "search implemented only for k=1");
    FAISS_THROW_IF_NOT_MSG (sum_d == d,
                      "not enough indexes compared to # dimensions");

    long nshard = sub_indexes.size();
    float *all_distances = new float [nshard * k * n];
    idx_t *all_labels = new idx_t [nshard * k * n];
    ScopeDeleter<float> del (all_distances);
    ScopeDeleter<idx_t> del2 (all_labels);

    // pre-alloc because we don't want reallocs
    std::vector<Thread<SplitQueryJob> > qss (nshard);
    for (int i = 0; i < nshard; i++) {
        SplitQueryJob qs = {
            this, i, n, x, k,
            i == 0 ? distances : all_distances + i * k * n,
            i == 0 ? labels : all_labels + i * k * n
        };
        if (threaded) {
            qss[i] = Thread<SplitQueryJob> (qs);
            qss[i].start();
        } else {
            qs.run();
        }
    }

    if (threaded) {
        for (int i = 0; i < qss.size(); i++) {
            qss[i].wait();
        }
    }

    long factor = 1;
    for (int i = 0; i < nshard; i++) {
        if (i > 0) { // results of 0 are already in the table
            const float *distances_i = all_distances + i * k * n;
            const idx_t *labels_i = all_labels + i * k * n;
            for (long j = 0; j < n; j++) {
                if (labels[j] >= 0 && labels_i[j] >= 0) {
                    labels[j] += labels_i[j] * factor;
                    distances[j] += distances_i[j];
                } else {
                    labels[j] = -1;
                    distances[j] = 0.0 / 0.0;
                }
            }
        }
        factor *= sub_indexes[i]->ntotal;
    }

}

void IndexSplitVectors::train(idx_t /*n*/, const float* /*x*/) {
  FAISS_THROW_MSG("not implemented");
}

void IndexSplitVectors::reset ()
{
    FAISS_THROW_MSG ("not implemented");
}


IndexSplitVectors::~IndexSplitVectors ()
{
    if (own_fields) {
        for (int s = 0; s < sub_indexes.size(); s++)
            delete sub_indexes [s];
    }
}






}; // namespace faiss
