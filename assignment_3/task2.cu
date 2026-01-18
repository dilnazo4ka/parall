#include <bits/stdc++.h>                 // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>                // подключаю CUDA Runtime API
using namespace std;                     // чтобы не писать std::

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA-вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печатаю ошибку */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* выхожу */ \
  } \
}while(0)

// kernel: поэлементное сложение двух векторов C = A + B
__global__ void vec_add(const float* a, const float* b, float* c, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n) c[idx] = a[idx] + b[idx];            // если в границе — складываю
}

// функция замера времени kernel через cudaEvent
static float time_kernel(function<void()> launch){
  cudaEvent_t ev_start, ev_end;                    // CUDA события
  CHECK(cudaEventCreate(&ev_start));               // создаю start event
  CHECK(cudaEventCreate(&ev_end));                 // создаю end event
  CHECK(cudaEventRecord(ev_start));                // старт
  launch();                                        // запуск kernel
  CHECK(cudaEventRecord(ev_end));                  // стоп
  CHECK(cudaEventSynchronize(ev_end));             // жду завершения
  float ms=0.0f;                                   // время в мс
  CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end)); // считаю время
  CHECK(cudaEventDestroy(ev_start));               // удаляю события
  CHECK(cudaEventDestroy(ev_end));
  return ms;                                       // возвращаю время
}

int main(){
  const int N = 1'000'000;                         // размер массивов
  vector<float> hA(N), hB(N), hC(N);               // host массивы

  mt19937 gen(42);                                 // генератор
  for(int i=0;i<N;i++){                            // заполняю входные данные
    hA[i] = float(gen()%100);                      // A[i]
    hB[i] = float(gen()%100);                      // B[i]
  }

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;     // указатели на GPU память
  CHECK(cudaMalloc(&dA, N*sizeof(float)));         // память под A
  CHECK(cudaMalloc(&dB, N*sizeof(float)));         // память под B
  CHECK(cudaMalloc(&dC, N*sizeof(float)));         // память под C

  CHECK(cudaMemcpy(dA, hA.data(),                  // копирую A на GPU
                   N*sizeof(float),
                   cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(dB, hB.data(),                  // копирую B на GPU
                   N*sizeof(float),
                   cudaMemcpyHostToDevice));

  vector<int> thread_options = {128, 256, 512};    // минимум 3 значения threads per block

  cout.setf(std::ios::fixed);                      // формат вывода
  cout << setprecision(6);                         // 6 знаков после запятой
  cout << "threads,blocks,time_ms,check\n";        // заголовок таблицы

  for(int threads : thread_options){               // перебираю размеры блока
    int blocks = (N + threads - 1) / threads;      // считаю количество блоков

    vec_add<<<blocks,threads>>>(dA,dB,dC,N);       // warm-up запуск
    CHECK(cudaDeviceSynchronize());                // синхронизация

    float ms = time_kernel([&](){                  // замер времени
      vec_add<<<blocks,threads>>>(dA,dB,dC,N);     // запуск kernel
      CHECK(cudaDeviceSynchronize());              // жду завершения
    });

    CHECK(cudaMemcpy(hC.data(), dC,                // копирую результат на CPU
                     N*sizeof(float),
                     cudaMemcpyDeviceToHost));

    bool ok = true;                                // флаг корректности
    for(int i=0;i<1000;i++){                       // проверяю первые 1000 элементов
      float ref = hA[i] + hB[i];                   // эталон
      if(fabs(hC[i]-ref) > 1e-4){ ok=false; break; }
    }

    cout << threads << ","                         // вывод threads
         << blocks << ","                          // вывод blocks
         << ms << ","                              // вывод time
         << (ok ? "OK" : "FAIL") << "\n";          // вывод проверки
  }

  cudaFree(dA);                                    // освобождаю память GPU
  cudaFree(dB);
  cudaFree(dC);
  return 0;                                        // конец
}
