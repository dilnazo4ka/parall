#include <bits/stdc++.h>                     // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>                    // подключаю CUDA Runtime API
using namespace std;                         // использую пространство имён std

#define CHECK(x) do {                        \
  cudaError_t err = (x);                     \
  if (err != cudaSuccess) {                  \
    printf("CUDA error: %s (%s:%d)\n",       \
           cudaGetErrorString(err),          \
           __FILE__, __LINE__);              \
    exit(1);                                 \
  }                                          \
} while (0)

__global__ void mul2_kernel(float* a, int n) {// CUDA kernel: умножение элементов на 2
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if (idx < n)                                      // проверяю границу массива
    a[idx] = a[idx] * 2.0f;                         // умножаю элемент на 2
}

int main(){                                        // точка входа
  const int N = 1'000'000;                          // размер массива
  const int CHUNK = 250'000;                        // размер одного чанка
  const int NSTREAMS = 4;                           // количество CUDA stream

  vector<float> h(N);                               // host массив
  for(int i = 0; i < N; i++)                        // инициализация массива
    h[i] = float(i);                                // заполняю значениями

  float* d[NSTREAMS];                               // массив указателей GPU памяти
  cudaStream_t streams[NSTREAMS];                   // массив CUDA stream

  for(int i = 0; i < NSTREAMS; i++){                // создаю stream и память
    CHECK(cudaStreamCreate(&streams[i]));           // создаю stream
    CHECK(cudaMalloc(&d[i], CHUNK * sizeof(float))); // выделяю память под чанк
  }

  int threads = 256;                                // потоки в блоке
  int blocks  = (CHUNK + threads - 1) / threads;    // блоки

  cudaEvent_t start, stop;                          // события для таймера
  CHECK(cudaEventCreate(&start));                   // создаю start event
  CHECK(cudaEventCreate(&stop));                    // создаю stop event

  CHECK(cudaEventRecord(start));                    // старт общего таймера

  for(int i = 0; i < NSTREAMS; i++){                // цикл по stream
    int offset = i * CHUNK;                         // смещение в массиве

    CHECK(cudaMemcpyAsync(d[i],                     // асинхронное копирование H→D
                           h.data() + offset,       // источник
                           CHUNK * sizeof(float),   // размер
                           cudaMemcpyHostToDevice,  // направление
                           streams[i]));            // stream

    mul2_kernel<<<blocks, threads, 0, streams[i]>>>(d[i], CHUNK); // запуск kernel в stream

    CHECK(cudaMemcpyAsync(h.data() + offset,        // асинхронное копирование D→H
                           d[i],                    // источник
                           CHUNK * sizeof(float),   // размер
                           cudaMemcpyDeviceToHost,  // направление
                           streams[i]));            // stream
  }

  CHECK(cudaEventRecord(stop));                     // фиксирую конец операций
  CHECK(cudaEventSynchronize(stop));                // жду завершения всех stream

  float ms = 0.0f;                                  // переменная времени
  CHECK(cudaEventElapsedTime(&ms, start, stop));    // вычисляю общее время

  bool ok = true;                                   // флаг проверки
  for(int i = 0; i < 1000; i++){                    // проверка корректности
    if(h[i] != float(i) * 2.0f){                    // сравниваю значение
      ok = false;                                   // если ошибка
      break;                                        // выхожу из цикла
    }
  }

  cout.setf(std::ios::fixed);                       // фиксированный формат вывода
  cout << setprecision(6);                          // точность вывода
  cout << "N=" << N << "\n";                         // размер массива
  cout << "streams=" << NSTREAMS << "\n";           // количество stream
  cout << "time_ms=" << ms << "\n";                 // общее время
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n"; // результат проверки

  for(int i = 0; i < NSTREAMS; i++){                // освобождаю ресурсы
    cudaFree(d[i]);                                 // освобождаю GPU память
    cudaStreamDestroy(streams[i]);                  // уничтожаю stream
  }

  cudaEventDestroy(start);                          // удаляю start event
  cudaEventDestroy(stop);                           // удаляю stop event

  return 0;                                         // корректный выход
}
