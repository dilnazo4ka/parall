#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
#include <omp.h>                         // OpenMP
using namespace std;                     // пространство имён std

// макрос проверки ошибок CUDA
#define CHECK(x) do{ \
  cudaError_t err__ = (x);               /* выполняю CUDA вызов */ \
  if(err__ != cudaSuccess){              /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать описания */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// CUDA kernel: умножение части массива на 2
__global__ void mul2_kernel(float* a, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n)                                      // проверка границы
    a[idx] = a[idx] * 2.0f;                        // обработка элемента
}

int main(){
  const int N = 1'000'000;                         // размер массива
  const int HALF = N / 2;                          // половина массива

  vector<float> h(N);                              // host массив
  for(int i = 0; i < N; i++)                       // инициализация
    h[i] = float(i);

  // ---------------- CPU часть (первая половина) ----------------
  double cpu_t1 = omp_get_wtime();                 // старт CPU таймера

  #pragma omp parallel for                         // OpenMP параллельный цикл
  for(int i = 0; i < HALF; i++){
    h[i] = h[i] * 2.0f;                            // обработка на CPU
  }

  double cpu_t2 = omp_get_wtime();                 // конец CPU таймера

  // ---------------- GPU часть (вторая половина) ----------------
  float* d_part = nullptr;                         // указатель GPU памяти
  CHECK(cudaMalloc(&d_part, HALF * sizeof(float))); // память под вторую половину

  CHECK(cudaMemcpy(d_part,                         // копирую вторую половину на GPU
                   h.data() + HALF,
                   HALF * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                               // потоки в блоке
  int blocks  = (HALF + threads - 1) / threads;    // количество блоков

  cudaEvent_t start, stop;                         // CUDA события
  CHECK(cudaEventCreate(&start));                  // start
  CHECK(cudaEventCreate(&stop));                   // stop

  CHECK(cudaEventRecord(start));                   // старт GPU таймера
  mul2_kernel<<<blocks, threads>>>(d_part, HALF);  // запуск kernel
  CHECK(cudaEventRecord(stop));                    // стоп GPU таймера
  CHECK(cudaEventSynchronize(stop));               // жду завершения

  float gpu_ms = 0.0f;                             // время GPU kernel
  CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

  CHECK(cudaEventDestroy(start));                  // удаляю события
  CHECK(cudaEventDestroy(stop));

  CHECK(cudaMemcpy(h.data() + HALF,                // копирую результат обратно
                   d_part,
                   HALF * sizeof(float),
                   cudaMemcpyDeviceToHost));

  cudaFree(d_part);                                // освобождаю GPU память

  // ---------------- Проверка корректности ----------------
  bool ok = true;
  for(int i = 0; i < 1000; i++){                   // проверяю начало массива
    if(h[i] != float(i) * 2.0f){ ok = false; break; }
  }
  for(int i = HALF; i < HALF + 1000; i++){         // проверяю часть GPU
    if(h[i] != float(i) * 2.0f){ ok = false; break; }
  }

  cout.setf(std::ios::fixed);
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "CPU(OpenMP) time = " << (cpu_t2 - cpu_t1) * 1000 << " ms\n";
  cout << "GPU(CUDA) kernel time = " << gpu_ms << " ms\n";
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";

  return 0;                                        // конец программы
}
