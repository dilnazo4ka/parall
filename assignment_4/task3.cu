#include <bits/stdc++.h>                     // стандартные C++ библиотеки
#include <cuda_runtime.h>                    // CUDA Runtime API
using namespace std;                         // пространство имён std

#define CHECK(x) do {                        \
  cudaError_t err = (x);                     \
  if (err != cudaSuccess) {                  \
    printf("CUDA error: %s (%s:%d)\n",       \
           cudaGetErrorString(err),          \
           __FILE__, __LINE__);              \
    exit(1);                                 \
  }                                          \
} while (0)

__global__ void square_kernel(const float* in, float* out, int n) { // CUDA kernel: возведение в квадрат
  int idx = blockIdx.x * blockDim.x + threadIdx.x;                  // глобальный индекс потока
  if (idx < n)                                                      // проверка границы
    out[idx] = in[idx] * in[idx];                                   // вычисляю квадрат
}

int main() {                                                        // точка входа
  const int N = 1'000'000;                                          // размер массива
  vector<float> h_in(N);                                            // входной массив CPU
  vector<float> h_out_cpu(N);                                       // результат CPU
  vector<float> h_out_gpu(N);                                       // результат GPU

  for (int i = 0; i < N; i++)                                       // инициализация входа
    h_in[i] = float(i % 100);                                       // заполняю значениями

  auto cpu_t1 = chrono::high_resolution_clock::now();               // старт CPU таймера
  for (int i = 0; i < N; i++)                                       // CPU вычисление
    h_out_cpu[i] = h_in[i] * h_in[i];                                // квадрат на CPU
  auto cpu_t2 = chrono::high_resolution_clock::now();               // конец CPU таймера
  double cpu_ms = chrono::duration<double, milli>(cpu_t2 - cpu_t1).count(); // CPU время

  float *d_in = nullptr;                                            // указатель GPU вход
  float *d_out = nullptr;                                           // указатель GPU выход
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));                      // выделяю память GPU
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));                     // выделяю память GPU

  CHECK(cudaMemcpy(d_in, h_in.data(),                               // копирую вход на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                                                // потоки в блоке
  int blocks  = (N + threads - 1) / threads;                        // число блоков

  cudaEvent_t s, e;                                                 // CUDA события
  CHECK(cudaEventCreate(&s));                                       // создаю start
  CHECK(cudaEventCreate(&e));                                       // создаю end

  CHECK(cudaEventRecord(s));                                        // старт GPU таймера
  square_kernel<<<blocks, threads>>>(d_in, d_out, N);               // запуск CUDA kernel
  CHECK(cudaEventRecord(e));                                        // конец GPU таймера
  CHECK(cudaEventSynchronize(e));                                   // жду завершения kernel

  float gpu_ms = 0.0f;                                              // переменная времени GPU
  CHECK(cudaEventElapsedTime(&gpu_ms, s, e));                       // получаю GPU время

  CHECK(cudaMemcpy(h_out_gpu.data(), d_out,                         // копирую результат GPU → CPU
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool ok = true;                                                   // флаг проверки
  for (int i = 0; i < 1000; i++) {                                  // проверяю первые элементы
    if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-5f) {                // сравнение CPU и GPU
      ok = false;                                                   // ошибка
      break;                                                        // выхожу
    }
  }

  cout.setf(std::ios::fixed);                                       // формат вывода
  cout << setprecision(6);
  cout << "N=" << N << "\n";                                        // размер задачи
  cout << "cpu_ms=" << cpu_ms << "\n";                              // время CPU
  cout << "gpu_ms=" << gpu_ms << "\n";                              // время GPU
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";                 // результат проверки

  cudaFree(d_in);                                                   // освобождаю GPU память
  cudaFree(d_out);                                                  // освобождаю GPU память
  cudaEventDestroy(s);                                              // удаляю события
  cudaEventDestroy(e);                                              // удаляю события
  return 0;                                                         // конец программы
}
