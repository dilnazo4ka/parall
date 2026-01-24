#include <bits/stdc++.h>                 // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>                // подключаю CUDA Runtime API
using namespace std;                     // использую пространство имён std

#define CHECK(x) do {                        \
  cudaError_t err = (x);                     \
  if (err != cudaSuccess) {                  \
    printf("CUDA error: %s (%s:%d)\n",       \
           cudaGetErrorString(err),          \
           __FILE__, __LINE__);              \
    exit(1);                                 \
  }                                          \
} while (0)

__global__ void sum_global(const float* in, float* out, int n) { // CUDA kernel суммирования
  int idx = blockIdx.x * blockDim.x + threadIdx.x;               // вычисляю глобальный индекс
  if (idx < n)                                                   // проверяю границу
    atomicAdd(out, in[idx]);                                     // атомарно добавляю элемент
}

int main() {
  const int N = 100000;                                          // размер массива
  vector<float> h(N);                                            // host массив

  for (int i = 0; i < N; i++)                                    // инициализация массива
    h[i] = 1.0f;                                                 // заполняю единицами

  double cpu_sum = 0.0;                                          // сумма на CPU
  auto t1 = chrono::high_resolution_clock::now();                // старт CPU таймера
  for (int i = 0; i < N; i++) cpu_sum += h[i];                   // считаю сумму
  auto t2 = chrono::high_resolution_clock::now();                // конец CPU таймера
  double cpu_ms = chrono::duration<double, milli>(t2 - t1).count(); // CPU время

  float *d_in = nullptr, *d_out = nullptr;                       // указатели GPU
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));                   // выделяю память под вход
  CHECK(cudaMalloc(&d_out, sizeof(float)));                      // память под результат
  CHECK(cudaMemcpy(d_in, h.data(),                               // копирую данные на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_out, 0, sizeof(float)));                    // обнуляю результат

  int threads = 256;                                             // потоки в блоке
  int blocks  = (N + threads - 1) / threads;                     // количество блоков

  cudaEvent_t s, e;                                              // CUDA события
  CHECK(cudaEventCreate(&s));                                    // создаю start
  CHECK(cudaEventCreate(&e));                                    // создаю end

  CHECK(cudaEventRecord(s));                                     // старт GPU таймера
  sum_global<<<blocks, threads>>>(d_in, d_out, N);               // запуск kernel
  CHECK(cudaEventRecord(e));                                     // конец таймера
  CHECK(cudaEventSynchronize(e));                                // жду завершения

  float gpu_ms = 0.0f;                                           // время GPU
  CHECK(cudaEventElapsedTime(&gpu_ms, s, e));                    // получаю время

  float gpu_sum = 0.0f;                                          // сумма GPU
  CHECK(cudaMemcpy(&gpu_sum, d_out,                              // копирую результат
                   sizeof(float),
                   cudaMemcpyDeviceToHost));

  cout.setf(std::ios::fixed);                                    // формат вывода
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "cpu_sum=" << cpu_sum << ", cpu_ms=" << cpu_ms << "\n";
  cout << "gpu_sum=" << gpu_sum << ", gpu_ms=" << gpu_ms << "\n";

  cudaFree(d_in);                                                // освобождаю память
  cudaFree(d_out);
  return 0;                                                      // конец программы
}
