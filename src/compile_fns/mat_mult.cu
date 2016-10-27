#include <cublas_v2.h>

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmult(const double *A, const double *B, double *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Do the actual multiplication
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  // Destroy the handle
  cublasDestroy(handle);
}
