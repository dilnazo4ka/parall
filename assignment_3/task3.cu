#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // чтобы не писать std::

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA-вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печатаю ошибку */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* выхожу */ \
  } \
}while(0)

// kernel (A): coalesced доступ — соседние потоки читают соседние элементы
__global__ void read_coalesced(const float* in, float* out, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
  if(idx < n) out[idx] = in[idx] * 1.000001f;       // простой доступ + лёгкая математика
}

// kernel (B): non-coalesced доступ — потоки читают данные “прыжками”
__global__ void read_noncoalesced(const float* in, float* out, int n, int stride){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // индекс потока
  int j = idx * stride;                             // индекс чтения с шагом stride
  if(j < n) out[j] = in[j] * 1.000001f;             // записываю в то же место
}

// замер времени kernel через cudaEvent
static float time_kernel(function<void()> launch){
  cudaEvent_t s,e;                                  // CUDA события
  CHECK(cudaEventCreate(&s));                       // start event
  CHECK(cudaEventCreate(&e));                       // end event
  CHECK(cudaEventRecord(s));                        // старт
  launch();                                         // запуск kernel
  CHECK(cudaEventRecord(e));                        // стоп
  CHECK(cudaEventSynchronize(e));                   // ожидание
  float ms=0.0f;                                    // время
  CHECK(cudaEventElapsedTime(&ms,s,e));             // считаю время
  CHECK(cudaEventDestroy(s));                       // удаляю события
  CHECK(cudaEventDestroy(e));
  return ms;                                        // возвращаю ms
}

int main(){
  const int N = 1'000'000;                          // размер массива
  vector<float> h(N);                               // host массив

  mt19937 gen(42);                                  // генератор
  for(int i=0;i<N;i++) h[i] = float(gen()%100);     // заполнение

  float *d_in=nullptr, *d_out=nullptr;              // указатели GPU
  CHECK(cudaMalloc(&d_in, N*sizeof(float)));        // память под вход
  CHECK(cudaMalloc(&d_out, N*sizeof(float)));       // память под выход

  CHECK(cudaMemcpy(d_in, h.data(),                  // копирую на GPU
                   N*sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                                // потоки в блоке
  int blocks  = (N + threads - 1) / threads;        // блоки

  // warm-up для coalesced
  read_coalesced<<<blocks,threads>>>(d_in,d_out,N); // warm-up запуск
  CHECK(cudaDeviceSynchronize());                   // синхронизация

  // замер coalesced
  float ms_coal = time_kernel([&](){
    read_coalesced<<<blocks,threads>>>(d_in,d_out,N); // coalesced kernel
    CHECK(cudaDeviceSynchronize());                   // жду завершения
  });

  // non-coalesced: задаю stride (чем больше, тем хуже coalescing)
  int stride = 32;                                  // шаг доступа (можешь менять: 2, 8, 32, 64)

  // warm-up для noncoalesced
  read_noncoalesced<<<blocks,threads>>>(d_in,d_out,N,stride); // warm-up
  CHECK(cudaDeviceSynchronize());                               // синхронизация

  // замер non-coalesced
  float ms_non = time_kernel([&](){
    read_noncoalesced<<<blocks,threads>>>(d_in,d_out,N,stride); // non-coalesced kernel
    CHECK(cudaDeviceSynchronize());                              // жду
  });

  // проверка корректности: сравню пару значений (не все, чтобы быстро)
  vector<float> out(N);                             // host output
  CHECK(cudaMemcpy(out.data(), d_out,               // копирую результат
                   N*sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool ok = true;                                   // флаг проверки
  for(int i=0;i<1000;i++){                          // проверяю первые 1000
    float ref = h[i] * 1.000001f;                   // эталон
    if(fabs(out[i] - ref) > 1e-2f){                 // погрешность побольше из-за float
      ok = false; break;
    }
  }

  cout.setf(std::ios::fixed);                       // формат вывода
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "threads=" << threads << ", blocks=" << blocks << "\n";
  cout << "coalesced_ms=" << ms_coal << "\n";
  cout << "noncoalesced_ms=" << ms_non << " (stride=" << stride << ")\n";
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";

  cudaFree(d_in);                                   // освобождаю память
  cudaFree(d_out);
  return 0;                                         // конец
}
