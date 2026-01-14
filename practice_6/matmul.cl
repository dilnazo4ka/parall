__kernel void matmul(                                     // объявляю OpenCL-ядро
    __global const float* A,                              // матрица A (N×M) в глобальной памяти
    __global const float* B,                              // матрица B (M×K) в глобальной памяти
    __global float* C,                                    // матрица C (N×K) в глобальной памяти
    const int N,                                          // число строк A и C
    const int M,                                          // число столбцов A и строк B
    const int K) {                                        // число столбцов B и C

    int row = get_global_id(0);                           // беру номер строки C
    int col = get_global_id(1);                           // беру номер столбца C

    if (row >= N || col >= K) return;                     // защита от выхода за границы

    float sum = 0.0f;                                     // локальная переменная (регистры) для суммы

    for (int t = 0; t < M; t++) {                         // пробегаю по общей размерности M
        sum += A[row * M + t] * B[t * K + col];           // C[row,col] += A[row,t] * B[t,col]
    }

    C[row * K + col] = sum;                               // записываю результат в C
}
