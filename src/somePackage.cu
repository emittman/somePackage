#include "header/iterator.h"
#include "header/max.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <iostream>

extern "C" SEXP Rcublas_max(SEXP x, SEXP n, SEXP dim){

  double *xptr = REAL(x);
  int N = INTEGER(n)[0], D = INTEGER(dim)[0];

  fvec_d dx(xptr, xptr+N*D);
  ivec_d dresult(N);

  cublas_max(dx, dresult, N, D);

  ivec_h hresult(N);
  thrust::copy(dresult.begin(), dresult.end(), hresult.begin());

  SEXP indices = PROTECT(allocVector(INTSXP, N));

  for(int i=0; i<N; ++i)
    INTEGER(indices)[i] = hresult[i];

  UNPROTECT(1);
  return indices;
}


