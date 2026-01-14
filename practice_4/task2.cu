#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

#define CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_LAST_KERNEL() do { \
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) { \
    printf("Kernel launch error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
  CHECK(cudaDeviceSynchronize()); \
} while(0)

// (a) global-only: каждый поток делает atomicAdd в один float
__global__ void sum_global_atomic(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(out, in[idx]);
}

// (b) shared reduce: редукция в shared + 1 atomicAdd на блок
__global__ void sum_shared_reduce(const float* in, float* out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  s[tid] = (idx < n) ? in[idx] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, s[0]);
}

static float run_kernel_global(const float* d_in, float* d_out, int n, int blocks, int threads) {
  CHECK(cudaMemset(d_out, 0, sizeof(float)));

  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  sum_global_atomic<<<blocks, threads>>>(d_in, d_out, n);
  CHECK(cudaEventRecord(e));

  CHECK_LAST_KERNEL();

  CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, s, e));

  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms;
}

static float run_kernel_shared(const float* d_in, float* d_out, int n, int blocks, int threads) {
  CHECK(cudaMemset(d_out, 0, sizeof(float)));

  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  sum_shared_reduce<<<blocks, threads, threads * (int)sizeof(float)>>>(d_in, d_out, n);
  CHECK(cudaEventRecord(e));

  CHECK_LAST_KERNEL();

  CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, s, e));

  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms;
}

int main() {
  const int N = 1'000'000;
  vector<float> h(N);

  mt19937 gen(42);
  for (int i = 0; i < N; i++) h[i] = float(gen() % 100);

  // CPU контроль суммы (чтобы понимать, что "должно быть")
  double cpu_sum = 0.0;
  for (int i = 0; i < N; i++) cpu_sum += h[i];

  float *d_in = nullptr, *d_out = nullptr;
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CHECK(cudaMalloc(&d_out, sizeof(float)));

  CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  // warm-up (чтобы убрать “холодный старт”)
  sum_global_atomic<<<blocks, threads>>>(d_in, d_out, N);
  CHECK_LAST_KERNEL();

  // (a)
  float ms_a = run_kernel_global(d_in, d_out, N, blocks, threads);
  float sum_a = 0.0f;
  CHECK(cudaMemcpy(&sum_a, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  // (b)
  float ms_b = run_kernel_shared(d_in, d_out, N, blocks, threads);
  float sum_b = 0.0f;
  CHECK(cudaMemcpy(&sum_b, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  cout.setf(std::ios::fixed);
  cout << setprecision(2);

  cout << "N=" << N << "\n";
  cout << "cpu_sum=" << cpu_sum << "\n";
  cout << "sum_global=" << sum_a << ", time_ms=" << ms_a << "\n";
  cout << "sum_shared=" << sum_b << ", time_ms=" << ms_b << "\n";

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
