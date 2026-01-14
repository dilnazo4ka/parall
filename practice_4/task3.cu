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
    printf("Kernel error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
  CHECK(cudaDeviceSynchronize()); \
} while(0)

static const int CHUNK = 256;     // bubble chunk (локальная память)
static const int TILE  = 256;     // merge tile (shared memory)

// --------------------
// bubble sort чанка в локальной памяти (1 поток на блок)
// --------------------
__global__ void bubble_sort_chunks(const int* in, int* out, int n) {
  int base = blockIdx.x * CHUNK;
  if (base >= n) return;

  int local[CHUNK];

  // один поток сортирует
  for (int i = 0; i < CHUNK; i++) {
    int idx = base + i;
    local[i] = (idx < n) ? in[idx] : INT_MAX;
  }

  for (int i = 0; i < CHUNK; i++) {
    for (int j = 0; j < CHUNK - 1 - i; j++) {
      if (local[j] > local[j + 1]) {
        int tmp = local[j];
        local[j] = local[j + 1];
        local[j + 1] = tmp;
      }
    }
  }

  for (int i = 0; i < CHUNK; i++) {
    int idx = base + i;
    if (idx < n) out[idx] = local[i];
  }
}

// --------------------
// merge двух соседних сегментов размера width
// shared memory используется как буфер TILE+TILE
// 1 поток на блок, чтобы не было дедлоков
// --------------------
__global__ void merge_pass_shared_1thread(const int* in, int* out, int width, int n) {
  extern __shared__ int sh[];
  int* L = sh;
  int* R = sh + TILE;

  int start = blockIdx.x * 2 * width;
  if (start >= n) return;

  int mid = min(start + width, n);
  int end = min(start + 2 * width, n);

  int i = start;
  int j = mid;
  int k = start;

  while (i < mid || j < end) {
    int lcount = 0;
    int rcount = 0;

    if (i < mid) {
      lcount = min(TILE, mid - i);
      for (int t = 0; t < lcount; t++) L[t] = in[i + t];
    }
    if (j < end) {
      rcount = min(TILE, end - j);
      for (int t = 0; t < rcount; t++) R[t] = in[j + t];
    }

    int li = 0, ri = 0;
    while (li < lcount && ri < rcount) {
      out[k++] = (L[li] <= R[ri]) ? L[li++] : R[ri++];
    }
    while (li < lcount) out[k++] = L[li++];
    while (ri < rcount) out[k++] = R[ri++];

    i += lcount;
    j += rcount;
  }
}

int main() {
  vector<int> sizes = {10000, 100000, 1000000};
  cout << "n,cpu_ms,gpu_ms\n";

  for (int n : sizes) {
    vector<int> h(n), h_cpu;
    mt19937 gen(42);
    for (int i = 0; i < n; i++) h[i] = gen() % 100000;
    h_cpu = h;

    // CPU sort
    auto t1 = chrono::high_resolution_clock::now();
    sort(h_cpu.begin(), h_cpu.end());
    auto t2 = chrono::high_resolution_clock::now();
    double cpu_ms = chrono::duration<double, milli>(t2 - t1).count();

    int *dA=nullptr, *dB=nullptr;
    CHECK(cudaMalloc(&dA, n * sizeof(int)));
    CHECK(cudaMalloc(&dB, n * sizeof(int)));
    CHECK(cudaMemcpy(dA, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s));

    // 1) bubble sort chunks: dA -> dB
    int chunk_blocks = (n + CHUNK - 1) / CHUNK;
    bubble_sort_chunks<<<chunk_blocks, 1>>>(dA, dB, n);
    CHECK_LAST_KERNEL();

    // 2) merge passes: width = CHUNK, 2*CHUNK, ...
    for (int width = CHUNK; width < n; width *= 2) {
      int pairs = (n + 2 * width - 1) / (2 * width);
      int shmem = 2 * TILE * (int)sizeof(int);
      merge_pass_shared_1thread<<<pairs, 1, shmem>>>(dB, dA, width, n); // 1 thread!
      CHECK_LAST_KERNEL();
      swap(dA, dB);
    }

    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, s, e));

    cudaFree(dA);
    cudaFree(dB);

    cout << n << "," << cpu_ms << "," << gpu_ms << "\n";
  }
  return 0;
}
