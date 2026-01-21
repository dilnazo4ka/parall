#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // пространство имён std

// макрос для проверки ошибок CUDA
#define CHECK(x) do{ \
  cudaError_t err__ = (x);               /* выполняю CUDA вызов */ \
  if(err__ != cudaSuccess){              /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печатаю описание */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// CUDA kernel: умножение каждого элемента массива на 2
__global__ void mul2_kernel(float* a, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n)                                      // проверка границы
    a[idx] = a[idx] * 2.0f;                        // обработка элемента
}

int main(){
  const int N = 1'000'000;                         // размер массива
  vector<float> h(N);                              // host массив

  // инициализация массива
  for(int i = 0; i < N; i++)
    h[i] = float(i);

  float* d_a = nullptr;                            // указатель на GPU массив
  CHECK(cudaMalloc(&d_a, N * sizeof(float)));      // выделяю память на GPU

  CHECK(cudaMemcpy(d_a, h.data(),                  // копирую данные на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                               // потоки в блоке
  int blocks  = (N + threads - 1) / threads;       // количество блоков

  // CUDA события для замера времени
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));                  // событие start
  CHECK(cudaEventCreate(&stop));                   // событие stop

  CHECK(cudaEventRecord(start));                   // старт таймера
  mul2_kernel<<<blocks, threads>>>(d_a, N);        // запуск kernel
  CHECK(cudaEventRecord(stop));                    // стоп таймера
  CHECK(cudaEventSynchronize(stop));               // жду завершения kernel

  float gpu_ms = 0.0f;                             // время выполнения kernel
  CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CHECK(cudaEventDestroy(start));                  // удаляю события
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaMemcpy(h.data(), d_a,                  // копирую результат обратно
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost));

  // проверка корректности
  bool ok = true;
  for(int i = 0; i < 1000; i++){
    if(h[i] != float(i) * 2.0f){ ok = false; break; }
  }

  cout.setf(std::ios::fixed);
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "GPU(CUDA) kernel time = " << gpu_ms << " ms\n";
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";

  cudaFree(d_a);                                   // освобождаю память GPU
  return 0;                                        // конец программы
}
