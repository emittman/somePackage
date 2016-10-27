#include "iterator.h"
#include "mat_mult.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <iostream>


extern "C" SEXP Rmat_mult(SEXP A, SEXP B, SEXP M, SEXP K, SEXP N){

  int m = INTEGER(M)[0], k = INTEGER(K)[0], n = INTEGER(N)[0];
  double *Aptr = REAL(A), *Bptr = REAL(B);
  fvec_d A_d(Aptr, Aptr + m*k);
  fvec_d B_d(Bptr, Bptr + k*n);
  fvec_d C_d(m*n);

  double *Adptr = thrust::raw_pointer_cast(&A_d.data());
  double *Bdptr = thrust::raw_pointer_cast(&B_d.data());
  double *Cdptr = thrust::raw_pointer_cast(&C_d.data());
  gpu_blas_mmult(Adptr, Bdptr, Cdptr, m, k, n);

  fvec_h C_h(m*n);
  thrust::copy(C_d.begin(), C_d.end(), C_h.begin());

  SEXP C = PROTECT(allocVector(REALSXP, m*n));

  for(int i=0; i<m*n; ++i)
    REAL(C)[i] = C_h[i];

  UNPROTECT(1);
  return C;
}


