#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

__device__ __forceinline__ void sift_down(int* a, int n, int i) {
  // просеивание вниз
  while (true) {
    int l = 2*i + 1;
    int r = 2*i + 2;
    int largest = i;

    if (l < n && a[l] > a[largest]) largest = l;
    if (r < n && a[r] > a[largest]) largest = r;

    if (largest == i) break;
    int tmp = a[i];
    a[i] = a[largest];
    a[largest] = tmp;
    i = largest;
  }
}
// разбиение на heaps и раздача для потоков
__global__ void heapify_range_kernel(int* a, int n, int start, int end_exclusive) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = start + tid;
  if (i >= end_exclusive) return;
  sift_down(a, n, i);
}
__global__ void extract_step_kernel(int* a, int last) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int tmp = a[0];
    a[0] = a[last];
    a[last] = tmp;
    sift_down(a, last, 0);
  }
}

static void fill_random(std::vector<int>& a, uint32_t seed=42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, 100);
  for (auto &x : a) x = dist(rng);
}

static void gpu_heap_sort(std::vector<int>& a) {
  int n = (int)a.size();
  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  int last_internal = n/2 - 1;
  if (last_internal >= 0) {
    // Нахождение высоты дерева и движение сверху вниз
    int max_k = 0;
    while (( (1 << (max_k+1)) - 2 ) <= last_internal) max_k++;

    for (int k = max_k; k >= 0; --k) {
      int level_start = (1 << k) - 1;
      int level_end   = (1 << (k+1)) - 1; // exclusive for internal candidates
      if (level_start > last_internal) continue;
      int start = level_start;
      int end_excl = std::min(level_end, last_internal + 1);
      int count = end_excl - start;

      int threads = 256;
      int blocks = (count + threads - 1) / threads;
      heapify_range_kernel<<<blocks, threads>>>(d, n, start, end_excl);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // извлекаем максимум
  for (int last = n - 1; last > 0; --last) {
    extract_step_kernel<<<1, 1>>>(d, last);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaMemcpy(a.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d);
}

static void cpu_heap_sort(std::vector<int>& a) {
  std::make_heap(a.begin(), a.end());
  std::sort_heap(a.begin(), a.end());
}

int main() {
  std::vector<int> sizes = {10000, 100000, 1000000};
  std::cout << "n,cpu_ms,gpu_ms\n";

  for (int n : sizes) {
    std::vector<int> base(n);
    fill_random(base, 123);

    std::vector<int> cpu = base;
    auto c1 = std::chrono::high_resolution_clock::now();
    cpu_heap_sort(cpu);
    auto c2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c2 - c1).count();

    std::vector<int> gpu = base;

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    gpu_heap_sort(gpu);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, s, e));


    std::cout << n << "," << cpu_ms << "," << gpu_ms << "\n";

    cudaEventDestroy(s);
    cudaEventDestroy(e);
  }

  return 0;
}
