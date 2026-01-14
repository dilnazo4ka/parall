#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA runtime API

using namespace std;

#define CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// =======================================================
// Очередь FIFO (как в методичке): data, head, tail, capacity
// enqueue: atomicAdd(tail)
// dequeue: atomicAdd(head) и проверка pos < tail
// =======================================================
struct Queue {
  int* data;           // массив данных (глобальная память)
  int head;            // индекс чтения (голова)
  int tail;            // индекс записи (хвост)
  int capacity;        // ёмкость очереди

  __device__ void init(int* buffer, int size) { // инициализация
    data = buffer;                              // привязываю буфер
    head = 0;                                   // голова = 0
    tail = 0;                                   // хвост = 0
    capacity = size;                            // сохраняю ёмкость
  }

  __device__ bool enqueue(int value) {          // добавить элемент в конец
    int pos = atomicAdd(&tail, 1);              // атомарно беру позицию для записи
    if (pos < capacity) {                       // если не вышли за ёмкость
      data[pos] = value;                        // пишу значение
      return true;                              // успех
    }
    return false;                               // очередь переполнена
  }

  __device__ bool dequeue(int* value) {         // удалить элемент из начала
    int pos = atomicAdd(&head, 1);              // атомарно беру позицию для чтения
    if (pos < tail) {                           // если позиция ещё "существует"
      *value = data[pos];                       // читаю значение
      return true;                              // успех
    }
    return false;                               // очередь пуста
  }
};

__global__ void kernel_init_queue(Queue* q, int* buf, int cap) {
  q->init(buf, cap);                            // инициализирую очередь
}

// kernel: параллельный enqueue каждый поток кладёт своё tid
__global__ void kernel_enqueue(Queue* q, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  // глобальный индекс потока
  if (tid < n) {                                    // защита по границе
    q->enqueue(tid);                                // кладу tid
  }
}

// kernel: параллельный dequeue каждый поток пытается достать 1 элемент
__global__ void kernel_dequeue(Queue* q, int* out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  // глобальный индекс
  if (tid < n) {
    int val;                                        // локальная переменная
    if (q->dequeue(&val)) out[tid] = val;           // если успешно — пишу значение
    else out[tid] = -1;                             // если пусто — -1
  }
}

// helper: замер kernel времени через cudaEvent
static float time_kernel(function<void()> launch) {
  cudaEvent_t s, e;                                 // события CUDA
  CHECK(cudaEventCreate(&s));                       // создаю start
  CHECK(cudaEventCreate(&e));                       // создаю end

  CHECK(cudaEventRecord(s));                        // старт
  launch();                                         // запускаю kernel(ы)
  CHECK(cudaEventRecord(e));                        // конец
  CHECK(cudaEventSynchronize(e));                   // жду завершения

  float ms = 0.0f;                                  // сюда время
  CHECK(cudaEventElapsedTime(&ms, s, e));           // считаю миллисекунды

  CHECK(cudaEventDestroy(s));                       // удаляю события
  CHECK(cudaEventDestroy(e));
  return ms;                                        // возвращаю время
}

int main() {
  const int CAPACITY = 1 << 20;                     // ёмкость очереди (1,048,576)
  const int N = 1 << 20;                            // количество операций enqueue/dequeue

  // выделяю память на GPU
  int* d_buffer = nullptr;                          // буфер очереди
  int* d_out = nullptr;                             // результаты dequeue
  Queue* d_queue = nullptr;                         // структура очереди

  CHECK(cudaMalloc(&d_buffer, CAPACITY * sizeof(int))); // data
  CHECK(cudaMalloc(&d_out, N * sizeof(int)));           // out
  CHECK(cudaMalloc(&d_queue, sizeof(Queue)));           // queue struct

  // инициализирую очередь
  kernel_init_queue<<<1,1>>>(d_queue, d_buffer, CAPACITY); // init
  CHECK(cudaDeviceSynchronize());                           // жду

  int threads = 256;                                       // потоки в блоке
  int blocks = (N + threads - 1) / threads;                // блоки

  // enqueue + время
  float enqueue_ms = time_kernel([&](){
    kernel_enqueue<<<blocks, threads>>>(d_queue, N);       // параллельный enqueue
    CHECK(cudaGetLastError());                             // проверяю launch
    CHECK(cudaDeviceSynchronize());                        // жду
  });

  // dequeue + время
  float dequeue_ms = time_kernel([&](){
    kernel_dequeue<<<blocks, threads>>>(d_queue, d_out, N); // параллельный dequeue
    CHECK(cudaGetLastError());                              // проверяю launch
    CHECK(cudaDeviceSynchronize());                         // жду
  });

  // копирую результат на CPU
  vector<int> out(N);                                      // host буфер
  CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost)); // копия

  // вывод: первые элементы
  // FIFO ожидаемо: сначала маленькие (0,1,2,...) если всё прошло ровно
  cout << "Queue FIFO dequeue (first 20):\n";
  for (int i = 0; i < 20; i++) cout << out[i] << " ";
  cout << "\n";

  // печать времени
  cout << "enqueue_ms=" << enqueue_ms << "\n";
  cout << "dequeue_ms=" << dequeue_ms << "\n";

  // очистка памяти
  cudaFree(d_buffer);
  cudaFree(d_out);
  cudaFree(d_queue);

  return 0;
}
