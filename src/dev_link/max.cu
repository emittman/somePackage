#include "../iterator.h"
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <cublas_v2.h>
#include <iostream>

typedef thrust::tuple<strideAccessor, intIter> my_tuple;

struct whichMax : thrust::unary_function<double, int>{
  int dim;
  __host__ __device__ whichMax(int dim): dim(dim){}

  __host__ __device__ int operator()(double &vec){

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int incx=1, n = dim, result =0;
    double *vec_ptr = thrust::raw_pointer_cast(&vec);
    //find the first index of a maximal element
    cublasIdamax(handle, n, vec_ptr, incx, &result);
    cublasDestroy_v2(handle);
    return result;
  }
};

void cublas_max(fvec_d &x, ivec_d &result, int n, int d){
  stride f(d);
  strideIter siter = thrust::transform_iterator<stride, countIter>(thrust::make_counting_iterator<int>(0), f);
  strideAccessor stridex = thrust::permutation_iterator<realIter, strideIter>(x.begin(), siter);

  whichMax g(d);

  //find the index of maximum for each of n subvectors
  thrust::copy(result.begin(), result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  thrust::transform(stridex, stridex + n, result.begin(),  g);
  thrust::copy(result.begin(), result.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
}
