__kernel void vector_add(__global const float* A,   // входной массив A в глобальной памяти
                         __global const float* B,   // входной массив B в глобальной памяти
                         __global float* C) {       // выходной массив C в глобальной памяти
    int id = get_global_id(0);                      // беру глобальный индекс потока
    C[id] = A[id] + B[id];                          // складываю элементы
}
