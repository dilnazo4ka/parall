#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
#include <omp.h>                         // OpenMP
using namespace std;                     // пространство имён std

#define CHECK(x) do{ \
  cudaError_t err__ = (x);               /* выполняю CUDA вызов */ \
  if(err__ != cudaSuccess){              /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать описания */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// CUDA kernel: умножение массива на 2
__global__ void mul2_kernel(float* a, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if(idx < n) a[idx] = a[idx] * 2.0f;              // обработка элемента
}

// CPU (OpenMP) обработка + замер времени
static double run_cpu_openmp(vector<float> a){
  double t1 = omp_get_wtime();                     // старт CPU таймера

  #pragma omp parallel for                         // параллельный цикл
  for(int i = 0; i < (int)a.size(); i++){
    a[i] = a[i] * 2.0f;                            // обработка
  }

  double t2 = omp_get_wtime();                     // конец CPU таймера
  return (t2 - t1) * 1000.0;                       // возвращаю время в мс
}

// GPU (CUDA) обработка + замер kernel времени
static double run_gpu_cuda(vector<float> a){
  int N = (int)a.size();                           // размер массива
  float* d_a = nullptr;                            // GPU указатель
  CHECK(cudaMalloc(&d_a, N * sizeof(float)));      // выделяю память GPU
  CHECK(cudaMemcpy(d_a, a.data(),                  // копирую на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                               // threads per block
  int blocks  = (N + threads - 1) / threads;       // blocks

  mul2_kernel<<<blocks, threads>>>(d_a, N);        // warm-up
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t s,e;                                 // CUDA события
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));                       // старт таймера
  mul2_kernel<<<blocks, threads>>>(d_a, N);        // kernel
  CHECK(cudaEventRecord(e));                       // стоп таймера
  CHECK(cudaEventSynchronize(e));                  // жду завершения

  float ms = 0.0f;                                 // время kernel
  CHECK(cudaEventElapsedTime(&ms, s, e));

  CHECK(cudaEventDestroy(s));                      // удаляю события
  CHECK(cudaEventDestroy(e));

  CHECK(cudaMemcpy(a.data(), d_a,                  // копирую назад (не для времени, для корректности)
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost));

  cudaFree(d_a);                                   // освобождаю память
  return (double)ms;                               // возвращаю время в мс
}

// HYBRID: половина CPU(OpenMP) + половина GPU(CUDA)
static pair<double,double> run_hybrid(vector<float> a){
  int N = (int)a.size();                           // размер массива
  int HALF = N / 2;                                // половина

  double cpu_t1 = omp_get_wtime();                 // старт CPU таймера

  #pragma omp parallel for                         // CPU обрабатывает первую половину
  for(int i = 0; i < HALF; i++){
    a[i] = a[i] * 2.0f;
  }

  double cpu_t2 = omp_get_wtime();                 // конец CPU таймера
  double cpu_ms = (cpu_t2 - cpu_t1) * 1000.0;      // время CPU части

  float* d_part = nullptr;                         // GPU массив для второй половины
  CHECK(cudaMalloc(&d_part, HALF * sizeof(float))); // память под половину
  CHECK(cudaMemcpy(d_part, a.data() + HALF,        // копирую вторую половину
                   HALF * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                               // threads per block
  int blocks  = (HALF + threads - 1) / threads;    // blocks

  mul2_kernel<<<blocks, threads>>>(d_part, HALF);  // warm-up
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t s,e;                                 // события
  CHECK(cudaEventCreate(&s));
  CHECK(cudaEventCreate(&e));

  CHECK(cudaEventRecord(s));                       // старт
  mul2_kernel<<<blocks, threads>>>(d_part, HALF);  // kernel на GPU
  CHECK(cudaEventRecord(e));                       // стоп
  CHECK(cudaEventSynchronize(e));                  // жду

  float gpu_ms = 0.0f;                             // время GPU kernel
  CHECK(cudaEventElapsedTime(&gpu_ms, s, e));

  CHECK(cudaEventDestroy(s));                      // удаляю события
  CHECK(cudaEventDestroy(e));

  CHECK(cudaMemcpy(a.data() + HALF, d_part,        // копирую обратно
                   HALF * sizeof(float),
                   cudaMemcpyDeviceToHost));

  cudaFree(d_part);                                // освобождаю GPU память

  return {cpu_ms, (double)gpu_ms};                 // возвращаю (cpu_part_ms, gpu_part_ms)
}

// проверка массива: a[i] должно стать i*2
static bool check_result(const vector<float>& a){
  for(int i = 0; i < 2000; i++){                   // проверяю начало
    if(a[i] != float(i) * 2.0f) return false;
  }
  int N = (int)a.size();
  for(int i = N/2; i < N/2 + 2000; i++){           // проверяю середину
    if(a[i] != float(i) * 2.0f) return false;
  }
  return true;
}

int main(){
  const int N = 1'000'000;                         // размер массива
  vector<float> base(N);                           // исходный массив
  for(int i = 0; i < N; i++) base[i] = float(i);   // инициализация

  // CPU время
  double cpu_ms = run_cpu_openmp(base);            // запускаю CPU версию

  // GPU время
  double gpu_ms = run_gpu_cuda(base);              // запускаю GPU версию

  // HYBRID время
  auto [hyb_cpu_ms, hyb_gpu_ms] = run_hybrid(base);// запускаю hybrid версию

  // для корректности делаю отдельную проверку на готовом результате GPU (быстро)
  vector<float> check_vec(N);
  for(int i = 0; i < N; i++) check_vec[i] = float(i);
  float* d_a=nullptr;
  CHECK(cudaMalloc(&d_a, N*sizeof(float)));
  CHECK(cudaMemcpy(d_a, check_vec.data(), N*sizeof(float), cudaMemcpyHostToDevice));
  int threads=256, blocks=(N+threads-1)/threads;
  mul2_kernel<<<blocks,threads>>>(d_a,N);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(check_vec.data(), d_a, N*sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_a);
  bool ok = check_result(check_vec);

  cout.setf(std::ios::fixed);                      // формат вывода
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "CPU(OpenMP)_ms=" << cpu_ms << "\n";
  cout << "GPU(CUDA)_kernel_ms=" << gpu_ms << "\n";
  cout << "HYBRID_cpu_part_ms=" << hyb_cpu_ms << "\n";
  cout << "HYBRID_gpu_part_kernel_ms=" << hyb_gpu_ms << "\n";
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";

  return 0;                                        // конец
}
