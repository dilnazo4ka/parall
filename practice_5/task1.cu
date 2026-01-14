#include <bits/stdc++.h>              // подключаю все стандартные C++ библиотеки (vector, iostream и т.д.)
#include <cuda_runtime.h>             // подключаю CUDA Runtime API для работы с GPU

using namespace std;                  // чтобы не писать std:: перед cout, vector и т.п.

// Макрос для проверки ошибок CUDA
#define CHECK(x) do {                                   \
  cudaError_t err = (x);                                /* выполняю CUDA-вызов и сохраняю код ошибки */ \
  if (err != cudaSuccess) {                             /* если произошла ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",                  /* печатаю текст ошибки */ \
           cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(1);                                            /* аварийно завершаю программу */ \
  }                                                     \
} while(0)

// Структура параллельного стека (LIFO)
struct Stack {
  int* data;        // указатель на массив данных стека в глобальной памяти GPU
  int top;          // индекс вершины стека
  int capacity;     // максимальное количество элементов в стеке

  // Инициализация стека
  __device__ void init(int* buffer, int size) {
    data = buffer;   // привязываю внешний буфер памяти к стеку
    top = -1;        // стек изначально пуст (вершина = -1)
    capacity = size; // сохраняю максимальный размер
  }

  // Операция push (добавление элемента)
  __device__ bool push(int value) {
    int pos = atomicAdd(&top, 1); // атомарно увеличиваю top и получаю позицию
    if (pos < capacity) {         // проверяю, не вышли ли за границы стека
      data[pos] = value;          // записываю значение в стек
      return true;                // операция успешна
    }
    return false;                 // стек переполнен
  }

  // Операция pop (удаление элемента)
  __device__ bool pop(int* value) {
    int pos = atomicSub(&top, 1); // атомарно уменьшаю top и получаю позицию
    if (pos >= 0) {               // если стек не был пуст
      *value = data[pos];         // извлекаю значение
      return true;                // операция успешна
    }
    return false;                 // стек пуст
  }
};

// CUDA kernel: инициализация стека
__global__ void kernel_init(Stack* s, int* buf, int cap) {
  s->init(buf, cap);              // вызываю device-функцию init
}

// CUDA kernel: параллельный push
__global__ void kernel_push(Stack* s, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // вычисляю глобальный индекс потока
  if (tid < n) {                                   // защита от выхода за границы
    s->push(tid);                                  // каждый поток кладёт своё число tid
  }
}

// CUDA kernel: параллельный pop
__global__ void kernel_pop(Stack* s, int* out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
  if (tid < n) {                                   // проверка границы
    int val;                                       // локальная переменная для значения
    if (s->pop(&val))                              // если удалось извлечь элемент
      out[tid] = val;                              // записываю результат
    else
      out[tid] = -1;                               // если стек пуст — пишу -1
  }
}

// =======================================================
// Главная функция
// =======================================================
int main() {
  const int CAPACITY = 1024;        // максимальная ёмкость стека
  const int N = 512;                // количество push/pop операций

  int* d_buffer;                    // буфер данных стека в GPU памяти
  int* d_out;                       // массив результатов pop
  Stack* d_stack;                   // структура стека в GPU памяти

  // Выделяю память на GPU
  CHECK(cudaMalloc(&d_buffer, CAPACITY * sizeof(int))); // память под данные стека
  CHECK(cudaMalloc(&d_out, N * sizeof(int)));           // память под результаты pop
  CHECK(cudaMalloc(&d_stack, sizeof(Stack)));           // память под структуру Stack

  // Инициализация стека
  kernel_init<<<1,1>>>(d_stack, d_buffer, CAPACITY);    // запускаю kernel init
  CHECK(cudaDeviceSynchronize());                        // жду завершения

  int threads = 256;                                     // количество потоков в блоке
  int blocks = (N + threads - 1) / threads;              // вычисляю число блоков

  // Параллельный push
  kernel_push<<<blocks, threads>>>(d_stack, N);          // каждый поток делает push
  CHECK(cudaDeviceSynchronize());                        // синхронизация

  // Параллельный pop
  kernel_pop<<<blocks, threads>>>(d_stack, d_out, N);    // каждый поток делает pop
  CHECK(cudaDeviceSynchronize());                        // синхронизация

  // Копирование результатов на CPU
  vector<int> out(N);                                    // host-массив
  CHECK(cudaMemcpy(out.data(), d_out,
                   N * sizeof(int),
                   cudaMemcpyDeviceToHost));             // копирую данные с GPU

  // Вывод результатов
  cout << "Popped values (first 20):\n";                 // заголовок
  for (int i = 0; i < 20; i++)                            // вывожу первые 20 значений
    cout << out[i] << " ";
  cout << "\n";

  // Освобождение памяти
  cudaFree(d_buffer);                                    // освобождаю буфер данных
  cudaFree(d_out);                                       // освобождаю буфер результатов
  cudaFree(d_stack);                                     // освобождаю структуру стека

  return 0;                                              // конец программы
}