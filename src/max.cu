#include "iterator.h"
#include <thrust/functional.h>
#include <cublas_v2.h>

typedef thrust::tuple<strideAccessor, intIter> my_tuple;
typedef thrust::zip_iterator<my_tuple> my_zip;
typedef thrust::tuple<double&, int&> el_tuple;

struct whichMax : thrust::unary_function<el_tuple &, void>{
  int dim;
  __host__ __device__ whichMax(int dim): dim(dim){}

  __device__ void operator()(el_tuple &Tup){

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    int incx=1, n = dim;
    double *x = thrust::raw_pointer_cast(&(thrust::get<0>(Tup)));
    double *result = thrust::raw_pointer_cast(&(thrust::get<1>(Tup)));
    //find the first index of a maximal element
    cublasIdamax(handle, x, incx, result)
    cublasDestroy_v2(handle);
  }
};

void cublas_max(fvec_d x, ivec_d result, int n, int d){
  stride f(d);
  strideIter siter = thrust::transform_iterator<stride, countIter>(thrust::make_counting_iterator<int>(0), f);
  strideAccessor stridex = thrust::permutation_iterator<realIter, strideIter>(x.begin(), siter);
  my_tuple tup = thrust::tuple<strideAccessor, intIter>(stridex, result.begin());
  my_zip zip = thrust::zip_iterator<my_tup>(tup);
  whichMax g(d);
  //find the index of maximum for each of n subvectors
  thrust::for_each(zip, zip + n, g);
}
