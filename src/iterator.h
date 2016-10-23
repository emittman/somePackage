#ifndef ITER_H
#define ITER_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

typedef thrust::device_vector<int> ivec_d;
typedef thrust::device_vector<double> fvec_d;
typedef thrust::device_vector<int>::iterator intIter;
typedef thrust::device_vector<double>::iterator realIter;
typedef thrust::host_vector<int> ivec_h;
typedef thrust::host_vector<double> fvec_h;

typedef thrust::counting_iterator<int> countIter;

//Used for generating rep( (1:len)*incr, times=infinity)
struct stride: public thrust::unary_function<int, int>{
  int incr;
  __host__ __device__ stride(int incr=1): incr(incr){}
  __host__ __device__ int operator()(int x){
    return x*incr;
  }
};

typedef thrust::transform_iterator<stride, countIter> strideIter;
typedef thrust::permutation_iterator<realIter, strideIter> strideAccessor;


#endif
