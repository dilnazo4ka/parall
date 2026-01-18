#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // пространство имён std

// макрос проверки ошибок CUDA
#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA вызов */ \
  if(err__!=cudaSuccess){                /* если произошла ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать ошибки */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// kernel (a): умножение массива на число через глобальную память
__global__ void mul_global(const float* in, float* out, float k, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n)                                      // проверка границы
    out[idx] = in[idx] * k;                        // умножаю элемент
}

// kernel (b): умножение с использованием shared memory
__global__ void mul_shared(const float* in, float* out, float k, int n){
  extern __shared__ float s[];                     // объявляю shared memory
  int tid = threadIdx.x;                           // индекс потока в блоке
  int idx = blockIdx.x * blockDim.x + tid;         // глобальный индекс элемента
  s[tid] = (idx < n) ? in[idx] : 0.0f;             // копирую данные в shared
  __syncthreads();                                 // синхронизация потоков
  if(idx < n)                                      // проверка границы
    out[idx] = s[tid] * k;                         // вычисляю результат
}

// функция замера времени kernel
static float time_kernel(function<void()> launch){
  cudaEvent_t ev_start, ev_end;                    // CUDA события
  CHECK(cudaEventCreate(&ev_start));               // создаю start event
  CHECK(cudaEventCreate(&ev_end));                 // создаю end event
  CHECK(cudaEventRecord(ev_start));                // старт таймера
  launch();                                        // запуск kernel
  CHECK(cudaEventRecord(ev_end));                  // стоп таймера
  CHECK(cudaEventSynchronize(ev_end));             // ожидание завершения
  float ms=0.0f;                                   // время выполнения
  CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end)); // вычисляю время
  CHECK(cudaEventDestroy(ev_start));               // удаляю события
  CHECK(cudaEventDestroy(ev_end));
  return ms;                                       // возвращаю время
}

int main(){
  const int N = 1'000'000;                         // размер массива
  const float K = 3.5f;                            // множитель

  vector<float> h(N);                              // host массив
  mt19937 gen(42);                                 // генератор случайных чисел
  for(int i=0;i<N;i++)                             // заполнение массива
    h[i] = float(gen()%100);

  float *d_in=nullptr, *d_out=nullptr;             // указатели на GPU память
  CHECK(cudaMalloc(&d_in, N*sizeof(float)));       // память под вход
  CHECK(cudaMalloc(&d_out, N*sizeof(float)));      // память под выход

  CHECK(cudaMemcpy(d_in, h.data(),                 // копирование на GPU
                   N*sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                               // потоки в блоке
  int blocks  = (N + threads - 1) / threads;       // количество блоков

  mul_global<<<blocks,threads>>>(d_in,d_out,K,N);  // warm-up запуск
  CHECK(cudaDeviceSynchronize());                  // синхронизация

  float ms_global = time_kernel([&](){             // замер global версии
    mul_global<<<blocks,threads>>>(d_in,d_out,K,N);
    CHECK(cudaDeviceSynchronize());
  });

  float ms_shared = time_kernel([&](){             // замер shared версии
    mul_shared<<<blocks,threads,
                threads*(int)sizeof(float)>>>(d_in,d_out,K,N);
    CHECK(cudaDeviceSynchronize());
  });

  vector<float> out(N);                            // host массив результата
  CHECK(cudaMemcpy(out.data(), d_out,              // копирую результат
                   N*sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool ok=true;                                    // флаг проверки
  for(int i=0;i<1000;i++){                         // проверяю часть элементов
    float ref = h[i]*K;
    if(fabs(out[i]-ref) > 1e-4){ ok=false; break; }
  }

  cout.setf(std::ios::fixed);                      // формат вывода
  cout<<setprecision(6);
  cout<<"N="<<N<<"\n";                             // размер данных
  cout<<"global_ms="<<ms_global<<"\n";             // время global версии
  cout<<"shared_ms="<<ms_shared<<"\n";             // время shared версии
  cout<<"check="<<(ok?"OK":"FAIL")<<"\n";          // корректность

  cudaFree(d_in);                                  // освобождаю память
  cudaFree(d_out);
  return 0;                                        // конец программы
}
