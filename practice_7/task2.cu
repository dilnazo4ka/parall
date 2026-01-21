#include <bits/stdc++.h>                 // стандартные C++ библиளம் библиотки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // пространство имён std

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать ошибки */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* аварийный выход */ \
  } \
}while(0)

// kernel: inclusive scan (prefix sum) внутри одного блока (для n <= blockDim.x)
__global__ void scan_inclusive_block(const int* in, int* out, int n){
  extern __shared__ int s[];                           // shared memory
  int tid = threadIdx.x;                               // индекс потока
  if(tid < n) s[tid] = in[tid];                        // загружаю вход в shared
  __syncthreads();                                     // синхронизация

  for(int offset = 1; offset < n; offset <<= 1){       // шаги scan: 1,2,4,...
    int val = 0;                                       // временное значение
    if(tid >= offset && tid < n) val = s[tid - offset];// беру элемент слева
    __syncthreads();                                   // чтобы все прочитали старые значения
    if(tid < n) s[tid] += val;                         // обновляю prefix sum
    __syncthreads();                                   // синхронизация после обновления
  }

  if(tid < n) out[tid] = s[tid];                       // записываю результат
}

// CPU scan для проверки
static vector<int> cpu_scan(const vector<int>& a){
  vector<int> pref(a.size());                          // выходной вектор
  int run = 0;                                         // текущая сумма
  for(size_t i=0;i<a.size();i++){                      // проход по массиву
    run += a[i];                                       // накапливаю сумму
    pref[i] = run;                                     // записываю prefix
  }
  return pref;                                         // возвращаю
}

int main(){
  const int N = 1024;                                  // размер тестового массива (влезает в 1 блок)
  vector<int> h(N);                                    // host вход
  for(int i=0;i<N;i++) h[i] = 1;                        // заполняю единицами (ожидаем 1,2,3,...)

  vector<int> ref = cpu_scan(h);                       // эталон на CPU

  int *d_in=nullptr, *d_out=nullptr;                   // указатели GPU
  CHECK(cudaMalloc(&d_in, N*sizeof(int)));             // память под вход
  CHECK(cudaMalloc(&d_out, N*sizeof(int)));            // память под выход
  CHECK(cudaMemcpy(d_in, h.data(),                     // копирую на GPU
                   N*sizeof(int),
                   cudaMemcpyHostToDevice));

  int threads = 1024;                                  // один блок на 1024 потока
  scan_inclusive_block<<<1, threads, threads*(int)sizeof(int)>>>(d_in, d_out, N); // запуск scan
  CHECK(cudaDeviceSynchronize());                      // жду завершения

  vector<int> out(N);                                  // host выход
  CHECK(cudaMemcpy(out.data(), d_out,                  // копирую результат
                   N*sizeof(int),
                   cudaMemcpyDeviceToHost));

  bool ok = true;                                      // флаг проверки
  for(int i=0;i<N;i++){                                // проверяю все элементы
    if(out[i] != ref[i]){ ok=false; break; }           // сравнение с CPU
  }

  cout << "N=" << N << "\n";                            // размер массива
  cout << "first10: ";                                  // первые 10 для наглядности
  for(int i=0;i<10;i++) cout << out[i] << " ";
  cout << "\n";
  cout << "last=" << out[N-1] << "\n";                  // последний элемент
  cout << "check=" << (ok ? "OK" : "FAIL") << "\n";     // результат проверки

  cudaFree(d_in);                                      // освобождаю память
  cudaFree(d_out);
  return 0;                                            // конец программы
}
