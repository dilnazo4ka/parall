#include <bits/stdc++.h>                 // подключаю все стандартные C++ библиотеки
#include <cuda_runtime.h>                // подключаю CUDA Runtime API
using namespace std;                     // чтобы не писать std::

// макрос для проверки ошибок CUDA-вызовов
#define CHECK(x) do { \
  cudaError_t err = (x);                 /* выполняю CUDA-функцию */ \
  if (err != cudaSuccess) {              /* если произошла ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* вывожу описание ошибки */ \
           cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1);                             /* завершаю программу */ \
  } \
} while(0)

// макрос для проверки запуска CUDA-ядра
#define CHECK_LAST_KERNEL() do { \
  cudaError_t err = cudaGetLastError();  /* получаю ошибку последнего kernel */ \
  if (err != cudaSuccess) {              /* если kernel запущен с ошибкой */ \
    printf("Kernel launch error: %s (%s:%d)\n", \
           cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
  CHECK(cudaDeviceSynchronize());        /* синхронизация устройства */ \
} while(0)

// вариант (a): каждый поток делает atomicAdd прямо в глобальную память
__global__ void sum_global_atomic(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // вычисляю глобальный индекс потока
  if (idx < n)                                     // проверяю границу массива
    atomicAdd(out, in[idx]);                       // атомарно прибавляю элемент к общей сумме
}

// вариант (b): редукция внутри блока в shared memory
__global__ void sum_shared_reduce(const float* in, float* out, int n) {
  extern __shared__ float s[];                     // объявляю shared memory массив
  int tid = threadIdx.x;                           // индекс потока внутри блока
  int idx = blockIdx.x * blockDim.x + tid;         // глобальный индекс элемента

  s[tid] = (idx < n) ? in[idx] : 0.0f;             // загружаю данные в shared memory
  __syncthreads();                                 // синхронизирую потоки блока

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) { // редукция в shared
    if (tid < stride)                            // только часть потоков работает
      s[tid] += s[tid + stride];                 // суммирую элементы
    __syncthreads();                             // синхронизация на каждом шаге
  }

  if (tid == 0)                                  // один поток на блок
    atomicAdd(out, s[0]);                        // атомарно добавляет сумму блока
}

// запуск и измерение времени варианта (a)
static float run_kernel_global(const float* d_in, float* d_out, int n, int blocks, int threads) {
  CHECK(cudaMemset(d_out, 0, sizeof(float)));     // обнуляю выходную сумму на GPU

  cudaEvent_t s, e;                               // CUDA события для тайминга
  CHECK(cudaEventCreate(&s));                     // создаю событие start
  CHECK(cudaEventCreate(&e));                     // создаю событие end

  CHECK(cudaEventRecord(s));                      // фиксирую старт
  sum_global_atomic<<<blocks, threads>>>(d_in, d_out, n); // запускаю kernel
  CHECK(cudaEventRecord(e));                      // фиксирую конец

  CHECK_LAST_KERNEL();                            // проверяю запуск ядра

  CHECK(cudaEventSynchronize(e));                 // жду завершения kernel
  float ms = 0.0f;                                // переменная для времени
  CHECK(cudaEventElapsedTime(&ms, s, e));         // считаю время в миллисекундах

  CHECK(cudaEventDestroy(s));                     // удаляю событие start
  CHECK(cudaEventDestroy(e));                     // удаляю событие end
  return ms;                                      // возвращаю время
}

// запуск и измерение времени варианта (b)
static float run_kernel_shared(const float* d_in, float* d_out, int n, int blocks, int threads) {
  CHECK(cudaMemset(d_out, 0, sizeof(float)));     // обнуляю выходную сумму

  cudaEvent_t s, e;                               // CUDA события
  CHECK(cudaEventCreate(&s));                     // start
  CHECK(cudaEventCreate(&e));                     // end

  CHECK(cudaEventRecord(s));                      // старт таймера
  sum_shared_reduce<<<blocks, threads, threads * (int)sizeof(float)>>>
      (d_in, d_out, n);                           // запуск kernel с shared memory
  CHECK(cudaEventRecord(e));                      // конец таймера

  CHECK_LAST_KERNEL();                            // проверка kernel

  CHECK(cudaEventSynchronize(e));                 // ожидание завершения
  float ms = 0.0f;                                // время выполнения
  CHECK(cudaEventElapsedTime(&ms, s, e));         // вычисляю время

  CHECK(cudaEventDestroy(s));                     // освобождаю события
  CHECK(cudaEventDestroy(e));
  return ms;                                      // возвращаю время
}

int main() {
  const int N = 1'000'000;                        // размер массива
  vector<float> h(N);                             // host-массив

  mt19937 gen(42);                                // генератор случайных чисел
  for (int i = 0; i < N; i++)
    h[i] = float(gen() % 100);                    // заполняю массив

  double cpu_sum = 0.0;                           // контрольная сумма на CPU
  for (int i = 0; i < N; i++)
    cpu_sum += h[i];                              // последовательное суммирование

  float *d_in = nullptr, *d_out = nullptr;        // указатели на GPU память
  CHECK(cudaMalloc(&d_in, N * sizeof(float)));    // память под входные данные
  CHECK(cudaMalloc(&d_out, sizeof(float)));       // память под сумму

  CHECK(cudaMemcpy(d_in, h.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));      // копирую данные на GPU

  int threads = 256;                              // потоки в блоке
  int blocks  = (N + threads - 1) / threads;      // количество блоков

  sum_global_atomic<<<blocks, threads>>>(d_in, d_out, N); // warm-up запуск
  CHECK_LAST_KERNEL();                            // проверка

  float ms_a = run_kernel_global(d_in, d_out, N, blocks, threads); // вариант (a)
  float sum_a = 0.0f;
  CHECK(cudaMemcpy(&sum_a, d_out,
                   sizeof(float),
                   cudaMemcpyDeviceToHost));      // копирую результат

  float ms_b = run_kernel_shared(d_in, d_out, N, blocks, threads); // вариант (b)
  float sum_b = 0.0f;
  CHECK(cudaMemcpy(&sum_b, d_out,
                   sizeof(float),
                   cudaMemcpyDeviceToHost));      // копирую результат

  cout.setf(std::ios::fixed);                      // фиксированный формат вывода
  cout << setprecision(2);                         // 2 знака после запятой

  cout << "N=" << N << "\n";                       // размер массива
  cout << "cpu_sum=" << cpu_sum << "\n";           // сумма на CPU
  cout << "sum_global=" << sum_a
       << ", time_ms=" << ms_a << "\n";            // результат и время варианта (a)
  cout << "sum_shared=" << sum_b
       << ", time_ms=" << ms_b << "\n";            // результат и время варианта (b)

  cudaFree(d_in);                                  // освобождаю GPU память
  cudaFree(d_out);
  return 0;                                        // конец программы
}