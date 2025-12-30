#include <bits/stdc++.h>      // подключаю стандартные C++ библиотеки
#include <cuda_runtime.h>    // подключаю CUDA Runtime API

// макрос для проверки ошибок CUDA
#define CHECK(x) if((x)!=cudaSuccess){printf("CUDA error\n"); exit(1);}

/*
  CUDA-ядро одного шага сортировки слиянием.
  Каждый поток:
  - обрабатывает два соседних отсортированных подмассива
  - выполняет их слияние
*/
__global__ void merge_step(int* in, int* out, int width, int n) {
    // вычисляю глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // определяю начало обрабатываемого сегмента
    int start = tid * 2 * width;
    if (start >= n) return;

    // вычисляю границы левой и правой частей
    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    // выполняю слияние
    int i = start, j = mid, k = start;
    while (i < mid && j < end)
        out[k++] = (in[i] <= in[j]) ? in[i++] : in[j++];
    while (i < mid) out[k++] = in[i++];
    while (j < end) out[k++] = in[j++];
}

// функция запуска merge sort на GPU
void gpu_merge_sort(std::vector<int>& a) {
    int n = a.size();

    // выделяю память на GPU
    int *d1, *d2;
    CHECK(cudaMalloc(&d1, n * sizeof(int)));
    CHECK(cudaMalloc(&d2, n * sizeof(int)));

    // копирую данные с CPU на GPU
    CHECK(cudaMemcpy(d1, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // итеративно увеличиваю размер сливаемых подмассивов
    for (int w = 1; w < n; w *= 2) {
        int blocks = (n + 2*w - 1) / (2*w);
        merge_step<<<blocks, 256>>>(d1, d2, w, n);
        std::swap(d1, d2);
    }

    // копирую результат обратно на CPU
    CHECK(cudaMemcpy(a.data(), d1, n * sizeof(int), cudaMemcpyDeviceToHost));

    // освобождаю память GPU
    cudaFree(d1);
    cudaFree(d2);
}

int main() {
    // размеры массивов для тестирования
    std::vector<int> sizes = {10000, 100000, 1000000};
    std::cout << "n,cpu_ms,gpu_ms\n";

    for (int n : sizes) {
        // создаю массив и его копию
        std::vector<int> a(n), b(n);

        // инициализирую данные
        std::mt19937 gen(42);
        for (int &x : a) x = gen() % 100;
        b = a;

        // ---------- CPU сортировка ----------
        auto t1 = std::chrono::high_resolution_clock::now();
        std::sort(b.begin(), b.end());
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ---------- GPU сортировка ----------
        cudaEvent_t s, e;
        cudaEventCreate(&s);
        cudaEventCreate(&e);

        cudaEventRecord(s);
        gpu_merge_sort(a);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float gpu;
        cudaEventElapsedTime(&gpu, s, e);

        // вывожу результаты
        std::cout << n << "," << cpu << "," << gpu << "\n";
    }
}
