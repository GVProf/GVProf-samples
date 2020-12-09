#include <cuda.h>
#include <cuda_runtime.h>

// read: single value-0 (50 % access)
// write: single value-1 (50 %access)
__global__
void op1(double *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N / 2) {
    p[idx] = p[idx] + 1;
  }
}

// zero value (25% access)
__global__
void op2(double *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N / 4) {
    p[idx] = 0.0;
  }
}

// approximate
__global__
void op3(double *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N / 2) {
    p[idx] = 1.0 + idx / 10000000.0;
  }
}

// dense value (50% access)
__global__
void op4(int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N / 2) {
    p[idx] = 3;
  }
}

// structured value (50% access)
__global__
void op5(int *p, size_t N) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx % 2 && idx < N) {
    p[idx] = idx;
  }
}

static const size_t N = 1000;
static const int THREADS = 128;

int main() {
  double *p1;
  cudaMalloc(&p1, N * sizeof(double));
  cudaMemset(&p1, 0, N * sizeof(double));

  auto blocks = (N - 1) / THREADS + 1;
  op1<<<blocks, THREADS>>>(p1, N);
  op2<<<blocks, THREADS>>>(p1, N);
  op3<<<blocks, THREADS>>>(p1, N);

  int *p2;
  cudaMalloc(&p2, N * sizeof(int));

  op4<<<blocks, THREADS>>>(p2, N);
  op5<<<blocks, THREADS>>>(p2, N);

  cudaFree(p1);
  cudaFree(p2);

  return 0;
}
