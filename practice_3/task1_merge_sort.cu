#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define CHECK(x) if((x)!=cudaSuccess){printf("CUDA error\n"); exit(1);}

__global__ void merge_step(int* in, int* out, int width, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 2 * width;
    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;
    while (i < mid && j < end)
        out[k++] = (in[i] <= in[j]) ? in[i++] : in[j++];
    while (i < mid) out[k++] = in[i++];
    while (j < end) out[k++] = in[j++];
}

void gpu_merge_sort(std::vector<int>& a) {
    int n = a.size();
    int *d1, *d2;
    CHECK(cudaMalloc(&d1, n * sizeof(int)));
    CHECK(cudaMalloc(&d2, n * sizeof(int)));
    CHECK(cudaMemcpy(d1, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    for (int w = 1; w < n; w *= 2) {
        int blocks = (n + 2*w - 1) / (2*w);
        merge_step<<<blocks, 256>>>(d1, d2, w, n);
        std::swap(d1, d2);
    }

    CHECK(cudaMemcpy(a.data(), d1, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d1); cudaFree(d2);
}

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};
    std::cout << "n,cpu_ms,gpu_ms\n";

    for (int n : sizes) {
        std::vector<int> a(n), b;
        std::mt19937 g(42);
        for (int& x : a) x = g() % 100;

        b = a;
        auto t1 = std::chrono::high_resolution_clock::now();
        std::sort(b.begin(), b.end());
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu = std::chrono::duration<double, std::milli>(t2 - t1).count();

        cudaEvent_t s,e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        gpu_merge_sort(a);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float gpu;
        cudaEventElapsedTime(&gpu, s, e);

        std::cout << n << "," << cpu << "," << gpu << "\n";
    }
}
