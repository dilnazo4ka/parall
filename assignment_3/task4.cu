#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // пространство имён std

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA-вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печатаю описание */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// kernel: простая поэлементная операция (C = A * 2 + 1)
__global__ void simple_kernel(const float* in, float* out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n) out[idx] = in[idx] * 2.0f + 1.0f;     // простая арифметика
}

// функция замера времени kernel
static float time_kernel(function<void()> launch){
  cudaEvent_t s,e;                                  // CUDA события
  CHECK(cudaEventCreate(&s));                       // start event
  CHECK(cudaEventCreate(&e));                       // end event
  CHECK(cudaEventRecord(s));                        // старт таймера
  launch();                                         // запуск kernel
  CHECK(cudaEventRecord(e));                        // стоп таймера
  CHECK(cudaEventSynchronize(e));                   // ожидание
  float ms=0.0f;                                    // время выполнения
  CHECK(cudaEventElapsedTime(&ms,s,e));             // считаю время
  CHECK(cudaEventDestroy(s));                       // удаляю события
  CHECK(cudaEventDestroy(e));
  return ms;                                        // возвращаю время
}

int main(){
  const int N = 1'000'000;                          // размер массива
  vector<float> h(N);                               // host массив

  mt19937 gen(42);                                  // генератор
  for(int i=0;i<N;i++) h[i] = float(gen()%100);     // заполнение массива

  float *d_in=nullptr, *d_out=nullptr;              // указатели GPU памяти
  CHECK(cudaMalloc(&d_in, N*sizeof(float)));        // память под вход
  CHECK(cudaMalloc(&d_out, N*sizeof(float)));       // память под выход

  CHECK(cudaMemcpy(d_in, h.data(),                  // копирую данные на GPU
                   N*sizeof(float),
                   cudaMemcpyHostToDevice));

  // ---------------- BAD CONFIG ----------------
  int bad_threads = 32;                             // маленький block size (неэффективно)
  int bad_blocks  = (N + bad_threads - 1) / bad_threads;

  simple_kernel<<<bad_blocks, bad_threads>>>(d_in,d_out,N); // warm-up
  CHECK(cudaDeviceSynchronize());

  float ms_bad = time_kernel([&](){                 // замер плохой конфигурации
    simple_kernel<<<bad_blocks, bad_threads>>>(d_in,d_out,N);
    CHECK(cudaDeviceSynchronize());
  });

  // ---------------- GOOD CONFIG ----------------
  int good_threads = 256;                            // оптимальный block size
  int good_blocks  = (N + good_threads - 1) / good_threads;

  simple_kernel<<<good_blocks, good_threads>>>(d_in,d_out,N); // warm-up
  CHECK(cudaDeviceSynchronize());

  float ms_good = time_kernel([&](){                // замер хорошей конфигурации
    simple_kernel<<<good_blocks, good_threads>>>(d_in,d_out,N);
    CHECK(cudaDeviceSynchronize());
  });

  // проверка корректности
  vector<float> out(N);                             // host результат
  CHECK(cudaMemcpy(out.data(), d_out,
                   N*sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool ok = true;                                   // флаг проверки
  for(int i=0;i<1000;i++){
    float ref = h[i]*2.0f + 1.0f;
    if(fabs(out[i]-ref) > 1e-4){ ok=false; break; }
  }

  cout.setf(std::ios::fixed);                       // формат вывода
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "bad_config:  threads=" << bad_threads
       << ", blocks=" << bad_blocks
       << ", time_ms=" << ms_bad << "\n";
  cout << "good_config: threads=" << good_threads
       << ", blocks=" << good_blocks
       << ", time_ms=" << ms_good << "\n";
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";

  cudaFree(d_in);                                   // освобождаю память
  cudaFree(d_out);
  return 0;                                         // конец программы
}
