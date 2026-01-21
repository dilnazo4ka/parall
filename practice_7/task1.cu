#include <bits/stdc++.h>                 // стандартные C++ библиотеки
#include <cuda_runtime.h>                // CUDA Runtime API
using namespace std;                     // чтобы не писать std::

#define CHECK(x) do{ \
  cudaError_t err__=(x);                 /* выполняю CUDA вызов */ \
  if(err__!=cudaSuccess){                /* если ошибка */ \
    printf("CUDA error: %s (%s:%d)\n",   /* печать ошибки */ \
           cudaGetErrorString(err__), __FILE__, __LINE__); \
    exit(1);                             /* выхожу */ \
  } \
}while(0)

// ядро редукции: каждый блок суммирует свой кусок массива и пишет 1 число в out[blockIdx.x]
__global__ void reduce_sum(const float* in, float* out, int n){
  extern __shared__ float s[];                        // shared memory для блока
  int tid = threadIdx.x;                              // индекс потока в блоке
  int idx = blockIdx.x * blockDim.x + tid;            // глобальный индекс элемента

  s[tid] = (idx < n) ? in[idx] : 0.0f;                // загружаю элемент в shared (или 0)
  __syncthreads();                                    // жду, пока все потоки загрузят

  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){ // редукция пополам
    if(tid < stride) s[tid] += s[tid + stride];       // суммирую пары
    __syncthreads();                                  // синхронизация после шага
  }

  if(tid == 0) out[blockIdx.x] = s[0];                // поток 0 пишет сумму блока
}

// CPU сумма для проверки
static double cpu_sum_vec(const vector<float>& a){
  double s = 0.0;                                     // double для точности
  for(float x : a) s += x;                            // суммирование
  return s;                                           // возвращаю сумму
}

int main(){
  const int Ntest = 1024;                             // тестовый размер (известная сумма)
  vector<float> h(Ntest);                             // host массив

  for(int i=0;i<Ntest;i++) h[i] = float(i + 1);       // заполняю 1,2,3,...,Ntest

  double cpu_sum = cpu_sum_vec(h);                    // считаю сумму на CPU
  double cpu_ref = (double)Ntest * (Ntest + 1) / 2;   // формула суммы 1..N

  float *d_in=nullptr, *d_partial=nullptr;            // указатели на GPU
  CHECK(cudaMalloc(&d_in, Ntest * sizeof(float)));    // память под вход
  CHECK(cudaMemcpy(d_in, h.data(),                    // копирую вход на GPU
                   Ntest * sizeof(float),
                   cudaMemcpyHostToDevice));

  int threads = 256;                                  // потоки в блоке
  int blocks  = (Ntest + threads - 1) / threads;      // блоки

  CHECK(cudaMalloc(&d_partial, blocks * sizeof(float))); // память под частичные суммы

  reduce_sum<<<blocks, threads, threads * (int)sizeof(float)>>>(d_in, d_partial, Ntest); // запуск
  CHECK(cudaDeviceSynchronize());                     // жду завершения

  vector<float> partial(blocks);                      // host для частичных сумм
  CHECK(cudaMemcpy(partial.data(), d_partial,         // копирую partial на CPU
                   blocks * sizeof(float),
                   cudaMemcpyDeviceToHost));

  double gpu_sum = 0.0;                               // финальная сумма на CPU
  for(float x : partial) gpu_sum += x;                // складываю суммы блоков

  cout.setf(std::ios::fixed);                         // формат вывода
  cout << setprecision(2);
  cout << "Ntest=" << Ntest << "\n";
  cout << "cpu_sum=" << cpu_sum << "\n";
  cout << "cpu_ref=" << cpu_ref << "\n";
  cout << "gpu_sum=" << gpu_sum << "\n";
  cout << "diff_cpu_ref=" << fabs(cpu_sum - cpu_ref) << "\n";
  cout << "diff_cpu_gpu=" << fabs(cpu_sum - gpu_sum) << "\n";

  cudaFree(d_in);                                     // освобождаю память
  cudaFree(d_partial);
  return 0;                                           // конец
}
