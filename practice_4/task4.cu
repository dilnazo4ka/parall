#include <bits/stdc++.h>
#include <cuda_runtime.h>

// Thrust для быстрой GPU-сортировки
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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
    printf("Kernel error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// CUDA events helper
static float elapsed_ms(cudaEvent_t s, cudaEvent_t e) {
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, s, e));
  return ms;
}

// TASK 2 part for TASK 4: SUM (a) global atomic (медленно)
__global__ void sum_global_atomic(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(out, in[idx]);
}

// TASK 2 optimized: SUM (b) shared reduce two-pass (быстро)
// 1-й pass: каждый блок считает частичную сумму в shared и пишет в partial[]
// 2-й pass: редукция partial[] до одного числа
__global__ void block_reduce_sum(const float* in, float* partial, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // загрузка в shared
  s[tid] = (idx < n) ? in[idx] : 0.0f;
  __syncthreads();

  // редукция в shared
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  // 0-й поток пишет сумму блока
  if (tid == 0) partial[blockIdx.x] = s[0];
}

__global__ void final_reduce_sum(const float* partial, float* out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  s[tid] = (idx < n) ? partial[idx] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0) out[blockIdx.x] = s[0];
}

// SORT on GPU: thrust::sort (очень быстрый)

// benchmark: device-only sum global atomic
static float bench_sum_global_atomic(const float* d_in, float* d_out, int n) {
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;

  CHECK(cudaMemset(d_out, 0, sizeof(float)));

  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));
  sum_global_atomic<<<blocks, threads>>>(d_in, d_out, n);
  CHECK_LAST_KERNEL();
  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));

  float ms = elapsed_ms(s, e);

  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms;
}

// benchmark: device-only sum shared two-pass
static float bench_sum_shared_two_pass(const float* d_in, float* d_partial, float* d_out, int n) {
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;

  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));

  // pass 1
  block_reduce_sum<<<blocks, threads, threads * (int)sizeof(float)>>>(d_in, d_partial, n);
  CHECK_LAST_KERNEL();

  // pass 2: редуцируем partial (размер = blocks) до одного числа
  // делаем в несколько итераций, пока не останется 1 элемент
  int cur_n = blocks;
  const float* cur_in = d_partial;
  float* cur_out = d_out; // будем “перетаскивать” в d_out, потом обратно

  while (cur_n > 1) {
    int b = (cur_n + threads - 1) / threads;
    final_reduce_sum<<<b, threads, threads * (int)sizeof(float)>>>(cur_in, cur_out, cur_n);
    CHECK_LAST_KERNEL();

    cur_n = b;
    cur_in = cur_out;

    // чтобы не аллоцировать третий буфер:
    // если следующий шаг нужен, пишем обратно в d_partial
    cur_out = (cur_out == d_out) ? d_partial : d_out;
  }

  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));

  float ms = elapsed_ms(s, e);

  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms;
}

// benchmark: CPU sort (std::sort)
static double bench_cpu_sort(vector<int>& a) {
  auto t1 = chrono::high_resolution_clock::now();
  sort(a.begin(), a.end());
  auto t2 = chrono::high_resolution_clock::now();
  return chrono::duration<double, milli>(t2 - t1).count();
}

// benchmark: GPU sort (thrust::sort) device-only
static float bench_gpu_sort_thrust(int* d_data, int n) {
  cudaEvent_t s, e;
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  thrust::device_ptr<int> begin = thrust::device_pointer_cast(d_data);
  thrust::device_ptr<int> end   = begin + n;

  CHECK(cudaEventRecord(s));
  thrust::sort(begin, end);                 // сортировка на GPU
  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));

  float ms = elapsed_ms(s, e);

  CHECK(cudaEventDestroy(s));
  CHECK(cudaEventDestroy(e));
  return ms;
}

int main() {
  // размеры из задания
  vector<int> sizes = {10000, 100000, 1000000};

  // для честного сравнения: фиксированный генератор
  mt19937 gen(42);

  // аллоцируем буферы под максимальный размер один раз
  int maxN = *max_element(sizes.begin(), sizes.end());

  // host buffers
  vector<float> h_sum(maxN);
  vector<int>   h_sort(maxN);

  // device buffers
  float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
  int   *d_sort = nullptr;

  CHECK(cudaMalloc(&d_in, maxN * sizeof(float)));
  // partial для редукции: максимум блоков = ceil(maxN/256)
  int maxBlocks = (maxN + 255) / 256;
  CHECK(cudaMalloc(&d_partial, maxBlocks * sizeof(float)));
  CHECK(cudaMalloc(&d_out, maxBlocks * sizeof(float))); // используем как рабочий буфер + финальный
  CHECK(cudaMalloc(&d_sort, maxN * sizeof(int)));

  // шапка CSV
  cout << "n,sum_global_atomic_ms,sum_shared_two_pass_ms,cpu_sort_ms,gpu_sort_thrust_ms\n";

  for (int n : sizes) {
    // генерю данные
    for (int i = 0; i < n; i++) {
      h_sum[i]  = float(gen() % 100);
      h_sort[i] = int(gen() % 100000);
    }

    // копирую на GPU
    CHECK(cudaMemcpy(d_in, h_sum.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sort, h_sort.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // SUM timings (device-only)
    float t_sum_global = bench_sum_global_atomic(d_in, d_out, n);
    float t_sum_shared = bench_sum_shared_two_pass(d_in, d_partial, d_out, n);

    // SORT timings
    // CPU sort на копии
    vector<int> cpu_copy(h_sort.begin(), h_sort.begin() + n);
    double t_cpu_sort = bench_cpu_sort(cpu_copy);

    // GPU sort device-only
    float t_gpu_sort = bench_gpu_sort_thrust(d_sort, n);

    // печать строкой CSV
    cout << n << ","
         << t_sum_global << ","
         << t_sum_shared << ","
         << t_cpu_sort << ","
         << t_gpu_sort << "\n";
  }

  cudaFree(d_in);
  cudaFree(d_partial);
  cudaFree(d_out);
  cudaFree(d_sort);
  return 0;
}
