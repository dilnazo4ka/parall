#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // пространство имён std

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать ошибки */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// kernel: редукция суммы (как в Task1), каждый блок пишет частичную сумму
__global__ void reduce_sum(const float* in, float* out, int n){
  extern __shared__ float s[];                       // shared memory
  int tid = threadIdx.x;                             // индекс потока в блоке
  int idx = blockIdx.x * blockDim.x + tid;           // глобальный индекс

  s[tid] = (idx < n) ? in[idx] : 0.0f;               // загружаю элемент или 0
  __syncthreads();                                   // синхронизация

  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if(tid < stride) s[tid] += s[tid + stride];      // суммирую пары
    __syncthreads();                                 // синхронизация
  }

  if(tid == 0) out[blockIdx.x] = s[0];               // пишу сумму блока
}

// CPU сумма + замер времени
static pair<double,double> cpu_sum_timed(const vector<float>& a){
  auto t1 = chrono::high_resolution_clock::now();    // старт таймера
  double s = 0.0;                                    // сумма
  for(float x : a) s += x;                           // суммирование
  auto t2 = chrono::high_resolution_clock::now();    // конец таймера
  double ms = chrono::duration<double, milli>(t2 - t1).count(); // время в мс
  return {s, ms};                                    // возвращаю (sum, ms)
}

// GPU сумма + замер времени (kernel only) через cudaEvent
static pair<double,float> gpu_sum_timed(const vector<float>& h){
  int n = (int)h.size();                             // размер массива
  float *d_in=nullptr, *d_partial=nullptr;           // указатели GPU

  CHECK(cudaMalloc(&d_in, n*sizeof(float)));         // память под вход
  CHECK(cudaMemcpy(d_in, h.data(),                   // копирую на GPU
                   n*sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                                 // потоки в блоке
  int blocks  = (n + threads - 1) / threads;         // блоки

  CHECK(cudaMalloc(&d_partial, blocks*sizeof(float))); // память под partial

  reduce_sum<<<blocks,threads,threads*sizeof(float)>>>(d_in,d_partial,n); // warm-up
  CHECK(cudaDeviceSynchronize());                    // синхронизация

  cudaEvent_t s,e;                                   // события CUDA
  CHECK(cudaEventCreate(&s));                        // start
  CHECK(cudaEventCreate(&e));                        // end

  CHECK(cudaEventRecord(s));                         // старт таймера
  reduce_sum<<<blocks,threads,threads*sizeof(float)>>>(d_in,d_partial,n); // kernel
  CHECK(cudaEventRecord(e));                         // стоп таймера
  CHECK(cudaEventSynchronize(e));                    // жду завершения

  float ms = 0.0f;                                   // время kernel
  CHECK(cudaEventElapsedTime(&ms, s, e));            // считаю ms

  CHECK(cudaEventDestroy(s));                        // удаляю события
  CHECK(cudaEventDestroy(e));

  vector<float> partial(blocks);                     // host partial суммы
  CHECK(cudaMemcpy(partial.data(), d_partial,        // копирую partial на CPU
                   blocks*sizeof(float),
                   cudaMemcpyDeviceToHost));

  double sum = 0.0;                                  // финальная сумма
  for(float x : partial) sum += x;                   // складываю partial на CPU

  cudaFree(d_in);                                    // освобождаю память
  cudaFree(d_partial);

  return {sum, ms};                                  // возвращаю (sum, kernel_ms)
}

int main(){
  const int N = 10'000'000;                           // большой размер для сравнения времени
  vector<float> h(N);                                 // host массив

  mt19937 gen(42);                                    // генератор
  for(int i=0;i<N;i++) h[i] = float(gen()%100);       // заполнение

  auto [cpu_sum, cpu_ms] = cpu_sum_timed(h);          // CPU сумма и время
  auto [gpu_sum, gpu_ms] = gpu_sum_timed(h);          // GPU сумма и время (kernel only)

  cout.setf(std::ios::fixed);                         // формат вывода
  cout << setprecision(2);

  cout << "N=" << N << "\n";                          // размер
  cout << "cpu_sum=" << cpu_sum << ", cpu_ms=" << cpu_ms << "\n"; // CPU
  cout << "gpu_sum=" << gpu_sum << ", gpu_kernel_ms=" << gpu_ms << "\n"; // GPU kernel
  cout << "diff=" << fabs(cpu_sum - gpu_sum) << "\n"; // разница

  return 0;                                           // конец
}
