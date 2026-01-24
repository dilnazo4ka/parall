#include <bits/stdc++.h>                                   // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>                                  // подключаю CUDA Runtime API
using namespace std;                                       // использую пространство имён std

#define CHECK(x) do {                                      \
  cudaError_t err = (x);                                   \
  if (err != cudaSuccess) {                                \
    printf("CUDA error: %s (%s:%d)\n",                     \
           cudaGetErrorString(err), __FILE__, __LINE__);   \
    exit(1);                                               \
  }                                                        \
} while (0)

__global__ void block_inclusive_scan(const float* in, float* out, float* block_sums, int n) { // kernel: inclusive scan внутри блока
  extern __shared__ float s[];                              // shared память для блока
  int tid = threadIdx.x;                                    // индекс потока в блоке
  int gid = blockIdx.x * blockDim.x + tid;                  // глобальный индекс элемента
  float x = (gid < n) ? in[gid] : 0.0f;                     // читаю элемент или 0 если вышли за границу
  s[tid] = x;                                               // кладу элемент в shared память
  __syncthreads();                                          // синхронизирую потоки

  for (int offset = 1; offset < blockDim.x; offset <<= 1) { // Hillis–Steele scan по степеням двойки
    float add = 0.0f;                                       // временная переменная для прибавки
    if (tid >= offset) add = s[tid - offset];               // беру значение на offset левее
    __syncthreads();                                        // синхронизирую перед записью
    s[tid] += add;                                          // обновляю префиксную сумму
    __syncthreads();                                        // синхронизирую после записи
  }

  if (gid < n) out[gid] = s[tid];                           // записываю результат для своего элемента
  if (tid == blockDim.x - 1) block_sums[blockIdx.x] = s[tid]; // последний поток пишет сумму блока
}

__global__ void add_block_offsets(float* out, const float* offsets, int n) { // kernel: добавляю оффсет блока ко всем элементам
  int gid = blockIdx.x * blockDim.x + threadIdx.x;          // глобальный индекс
  if (gid < n) out[gid] += offsets[blockIdx.x];             // добавляю смещение для блока
}

int main() {                                                // точка входа
  const int N = 1'000'000;                                  // размер массива по заданию
  vector<float> h_in(N);                                    // входной массив на CPU
  for (int i = 0; i < N; i++) h_in[i] = 1.0f;               // заполняю единицами (чтобы легко проверять)

  vector<float> cpu_out(N);                                 // результат CPU scan
  auto cpu_t1 = chrono::high_resolution_clock::now();       // старт CPU таймера
  float run = 0.0f;                                         // накопитель для префиксной суммы
  for (int i = 0; i < N; i++) {                             // цикл по массиву
    run += h_in[i];                                         // накапливаю сумму
    cpu_out[i] = run;                                       // записываю inclusive scan
  }
  auto cpu_t2 = chrono::high_resolution_clock::now();       // конец CPU таймера
  double cpu_ms = chrono::duration<double, milli>(cpu_t2 - cpu_t1).count(); // CPU время

  float *d_in = nullptr;                                    // указатель на вход на GPU
  float *d_out = nullptr;                                   // указатель на выход на GPU
  float *d_block_sums = nullptr;                            // суммы блоков на GPU
  float *d_offsets = nullptr;                               // оффсеты блоков на GPU

  CHECK(cudaMalloc(&d_in, N * sizeof(float)));              // выделяю память под вход
  CHECK(cudaMalloc(&d_out, N * sizeof(float)));             // выделяю память под выход

  int threads = 256;                                        // число потоков в блоке
  int blocks = (N + threads - 1) / threads;                 // число блоков

  CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float))); // выделяю память под суммы блоков
  CHECK(cudaMalloc(&d_offsets, blocks * sizeof(float)));    // выделяю память под оффсеты блоков

  CHECK(cudaMemcpy(d_in, h_in.data(),                       // копирую вход на GPU
                   N * sizeof(float),
                   cudaMemcpyHostToDevice));

  cudaEvent_t s, e;                                         // события CUDA для измерения времени
  CHECK(cudaEventCreate(&s));                               // создаю start event
  CHECK(cudaEventCreate(&e));                               // создаю end event

  CHECK(cudaEventRecord(s));                                // старт GPU таймера

  block_inclusive_scan<<<blocks, threads, threads * (int)sizeof(float)>>>( // запускаю scan внутри блоков
      d_in, d_out, d_block_sums, N);

  CHECK(cudaEventRecord(e));                                // конец измерения для 1-го kernel
  CHECK(cudaEventSynchronize(e));                           // жду завершения kernel

  vector<float> h_block_sums(blocks);                       // буфер на CPU для сумм блоков
  CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,       // копирую суммы блоков на CPU
                   blocks * sizeof(float),
                   cudaMemcpyDeviceToHost));

  vector<float> h_offsets(blocks);                          // буфер оффсетов на CPU
  float acc = 0.0f;                                         // накопитель оффсетов
  for (int b = 0; b < blocks; b++) {                        // считаю эксклюзивный scan по блокам
    h_offsets[b] = acc;                                     // оффсет блока = сумма всех предыдущих блоков
    acc += h_block_sums[b];                                 // обновляю накопитель
  }

  CHECK(cudaMemcpy(d_offsets, h_offsets.data(),             // копирую оффсеты обратно на GPU
                   blocks * sizeof(float),
                   cudaMemcpyHostToDevice));

  CHECK(cudaEventRecord(s));                                // старт замера для 2-го kernel

  add_block_offsets<<<blocks, threads>>>(d_out, d_offsets, N); // добавляю оффсеты к каждому элементу

  CHECK(cudaEventRecord(e));                                // конец замера для 2-го kernel
  CHECK(cudaEventSynchronize(e));                           // жду завершения kernel

  float gpu_ms = 0.0f;                                      // переменная для времени GPU
  CHECK(cudaEventElapsedTime(&gpu_ms, s, e));               // получаю время 2-го этапа (add offsets)

  vector<float> gpu_out(N);                                 // выходной массив на CPU
  CHECK(cudaMemcpy(gpu_out.data(), d_out,                   // копирую результат scan с GPU на CPU
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost));

  bool ok = true;                                           // флаг корректности
  for (int i = 0; i < 1000; i++) {                          // проверяю первые 1000 элементов
    if (fabs(cpu_out[i] - gpu_out[i]) > 1e-3f) {            // сравниваю CPU и GPU
      ok = false;                                           // если разница большая — ошибка
      break;                                                // выхожу
    }
  }

  cout.setf(std::ios::fixed);                               // фиксированный формат вывода
  cout << setprecision(6);                                  // точность
  cout << "N=" << N << "\n";                                // вывожу размер
  cout << "cpu_ms=" << cpu_ms << "\n";                      // вывожу время CPU
  cout << "gpu_ms_add_offsets=" << gpu_ms << "\n";          // вывожу время GPU для 2-го kernel
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";         // вывожу результат проверки
  cout << "cpu_last=" << cpu_out[N-1] << "\n";              // показываю последний элемент CPU
  cout << "gpu_last=" << gpu_out[N-1] << "\n";              // показываю последний элемент GPU

  cudaFree(d_in);                                           // освобождаю память GPU
  cudaFree(d_out);                                          // освобождаю память GPU
  cudaFree(d_block_sums);                                   // освобождаю память GPU
  cudaFree(d_offsets);                                      // освобождаю память GPU
  cudaEventDestroy(s);                                      // удаляю событие
  cudaEventDestroy(e);                                      // удаляю событие
  return 0;                                                 // конец программы
}
