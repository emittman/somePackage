#include "iterator.h"
#include "mat_mult.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <iostream>


extern "C" SEXP Rmat_mult(SEXP A, SEXP B){

  int m = nrow(A), k = nrow(B), n = ncol(B);
  double *Aptr = REAL(A), *Bptr = REAL(B);
  fvec_d A_d(Aptr, Aptr + m*k);
  fvec_d B_d(Bptr, Bptr + k*n);
  fvec_d C_d(m*n);

  gpu_blas_mmult(thrust::raw_pointer_cast(&A_d.data()),
                 thrust::raw_pointer_cast(&B_d.data()),,
                 thrust::raw_pointer_cast(&C_d.data()),
                 m, k, n);

  fvec_h C_h(m*n);
  thrust::copy(C_d.begin(), C_d.end(), C_h.begin());

  SEXP C = PROTECT(allocVector(REALSXP, m*n));

  for(int i=0; i<m*n; ++i)
    REAL(C)[i] = C_h[i];

  UNPROTECT(1);
  return C;
}


