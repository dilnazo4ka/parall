#include <bits/stdc++.h>                    // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>                   // подключаю CUDA Runtime API
using namespace std;                        // использую пространство имён std

#define CHECK(x) do {                        \
  cudaError_t err = (x);                     \
  if (err != cudaSuccess) {                  \
    printf("CUDA error: %s (%s:%d)\n",       \
           cudaGetErrorString(err),          \
           __FILE__, __LINE__);              \
    exit(1);                                 \
  }                                          \
} while (0)

__global__ void coalesced_kernel(float* a, int n){ // kernel с coalesced доступом
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисляю глобальный индекс
  if(idx < n)                                      // проверяю границу массива
    a[idx] = a[idx] * 2.0f;                        // обращение к соседним элементам
}

__global__ void noncoalesced_kernel(float* a, int n){ // kernel с non-coalesced доступом
  int idx = blockIdx.x * blockDim.x + threadIdx.x;    // вычисляю глобальный индекс
  int j = (idx * 32) % n;                             // искусственно нарушаю порядок доступа
  if(idx < n)                                         // проверяю границу
    a[j] = a[j] * 2.0f;                               // некоалесцированный доступ
}

float run_kernel(void(*kernel)(float*,int), float* d_a, int n, int blocks, int threads){
  cudaEvent_t s, e;                                  // CUDA события для таймера
  CHECK(cudaEventCreate(&s));                        // создаю start событие
  CHECK(cudaEventCreate(&e));                        // создаю end событие

  CHECK(cudaEventRecord(s));                         // записываю начало измерения
  kernel<<<blocks, threads>>>(d_a, n);               // запускаю kernel
  CHECK(cudaEventRecord(e));                         // записываю конец измерения
  CHECK(cudaEventSynchronize(e));                    // жду завершения kernel

  float ms = 0.0f;                                   // переменная для времени
  CHECK(cudaEventElapsedTime(&ms, s, e));            // получаю время выполнения

  CHECK(cudaEventDestroy(s));                        // удаляю start событие
  CHECK(cudaEventDestroy(e));                        // удаляю end событие
  return ms;                                         // возвращаю время
}

int main(){
  const int N = 1'000'000;                            // размер массива
  vector<float> h(N);                                // host массив
  for(int i = 0; i < N; i++)                         // инициализация массива
    h[i] = float(i);

  float* d_a = nullptr;                              // указатель на GPU массив
  CHECK(cudaMalloc(&d_a, N * sizeof(float)));        // выделяю память на GPU
  CHECK(cudaMemcpy(d_a, h.data(),                    // копирую данные на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                                 // количество потоков в блоке
  int blocks  = (N + threads - 1) / threads;         // количество блоков

  coalesced_kernel<<<blocks, threads>>>(d_a, N);     // прогрев GPU
  CHECK(cudaDeviceSynchronize());                    // синхронизация

  float t_coal = run_kernel(coalesced_kernel, d_a, N, blocks, threads); // coalesced время
  float t_non  = run_kernel(noncoalesced_kernel, d_a, N, blocks, threads); // non-coalesced время

  cout.setf(std::ios::fixed);                        // фиксированный формат вывода
  cout << setprecision(6);                           // точность вывода
  cout << "N=" << N << "\n";                          // вывожу размер массива
  cout << "coalesced_ms=" << t_coal << "\n";         // время coalesced доступа
  cout << "noncoalesced_ms=" << t_non << "\n";       // время non-coalesced доступа
  cout << "slowdown=" << t_non / t_coal << "\n";     // замедление

  cudaFree(d_a);                                     // освобождаю GPU память
  return 0;                                          // завершаю программу
}
