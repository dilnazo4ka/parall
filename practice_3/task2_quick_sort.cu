#include <bits/stdc++.h>
#include <cuda_runtime.h>

__global__ void partition_kernel(int* data, int pivot, int* less, int* greater, int n, int* lc, int* gc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (data[i] < pivot)
        less[atomicAdd(lc,1)] = data[i];
    else
        greater[atomicAdd(gc,1)] = data[i];
}

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};
    std::cout << "n,cpu_ms,gpu_partition_ms\n";

    for (int n : sizes) {
        std::vector<int> a(n);
        std::mt19937 g(1);
        for (int& x : a) x = g() % 100;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::sort(a.begin(), a.end());
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu = std::chrono::duration<double, std::milli>(t2 - t1).count();

        int *d, *l, *gr, *lc, *gc;
        cudaMalloc(&d,n*sizeof(int));
        cudaMalloc(&l,n*sizeof(int));
        cudaMalloc(&gr,n*sizeof(int));
        cudaMalloc(&lc,sizeof(int));
        cudaMalloc(&gc,sizeof(int));
        cudaMemcpy(d,a.data(),n*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemset(lc,0,sizeof(int));
        cudaMemset(gc,0,sizeof(int));

        cudaEvent_t s,e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        partition_kernel<<<(n+255)/256,256>>>(d,50,l,gr,n,lc,gc);
        cudaEventRecord(e);
        cudaEventSynchronize(e);

        float gpu;
        cudaEventElapsedTime(&gpu,s,e);
        std::cout << n << "," << cpu << "," << gpu << "\n";
    }
}
