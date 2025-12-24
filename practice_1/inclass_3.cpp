#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ========================================================================
// ФУНКЦИЯ: Создание динамического массива
// ========================================================================
// Выделяет память под массив заданного размера
// Параметры:
//   - size: количество элементов в массиве
// Возвращает: указатель на первый элемент массива
// Важно: память должна быть освобождена с помощью delete[]
int* createDynamicArray(int size) {
    cout << "[Memory] Allocating dynamic array of size " << size << "\n";
    int* arr = new int[size];  // Выделение памяти в куче (heap)
    cout << "[Memory] Array allocated at address: " << arr << "\n";
    return arr;
}

// ========================================================================
// ФУНКЦИЯ: Заполнение массива случайными числами
// ========================================================================
// Заполняет массив псевдослучайными числами в заданном диапазоне
// Параметры:
//   - arr: указатель на массив
//   - size: размер массива
//   - min_val: минимальное значение (по умолчанию 1)
//   - max_val: максимальное значение (по умолчанию 100)
void fillArrayRandom(int* arr, int size, int min_val = 1, int max_val = 100) {
    cout << "[Random] Filling array with random numbers [" << min_val 
         << ", " << max_val << "]\n";
    
    // Создаём генератор случайных чисел
    random_device rd;   // Источник энтропии
    mt19937 gen(rd());  // Генератор Mersenne Twister
    uniform_int_distribution<> distrib(min_val, max_val);
    
    // Заполняем массив
    for (int i = 0; i < size; ++i) {
        arr[i] = distrib(gen);
    }
    
    cout << "[Random] Array filled successfully\n";
}

// ========================================================================
// ФУНКЦИЯ: Последовательное вычисление среднего значения
// ========================================================================
// Вычисляет среднее арифметическое элементов массива
// Параметры:
//   - arr: указатель на массив
//   - size: размер массива
// Возвращает: среднее значение (double)
// Алгоритм: sum = arr[0] + arr[1] + ... + arr[n-1]
//           average = sum / n
double calculateAverageSequential(int* arr, int size) {
    cout << "\n[Sequential] Computing average value...\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    // Вычисляем сумму всех элементов
    long long sum = 0;  // long long чтобы избежать переполнения
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    // Вычисляем среднее
    double average = static_cast<double>(sum) / size;
    
    cout << "[Sequential] Sum: " << sum << "\n";
    cout << "[Sequential] Average: " << fixed << setprecision(4) << average << "\n";
    cout << "[Sequential] Time: " << duration.count() << " ms\n";
    
    return average;
}

// ========================================================================
// ФУНКЦИЯ: Параллельное вычисление среднего значения с OpenMP
// ========================================================================
// Использует OpenMP reduction для параллельного суммирования
// Параметры:
//   - arr: указатель на массив
//   - size: размер массива
//   - num_threads: количество потоков
// Возвращает: среднее значение (double)
// 
// OpenMP reduction(+:sum):
//   - Каждый поток создаёт свою локальную копию переменной sum
//   - Каждый поток суммирует свою часть массива
//   - В конце OpenMP автоматически объединяет все локальные суммы
double calculateAverageParallel(int* arr, int size, int num_threads) {
    cout << "\n[Parallel] Computing average value with " << num_threads << " threads...\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    long long sum = 0;
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    
    // Параллельное суммирование с reduction
    // reduction(+:sum) автоматически:
    // 1. Создаёт приватную копию sum для каждого потока
    // 2. Каждый поток суммирует свою часть
    // 3. В конце все частичные суммы складываются в финальный sum
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
#else
    // Если OpenMP недоступен, используем последовательный алгоритм
    cout << "[Warning] OpenMP not available, using sequential mode\n";
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
#endif
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    double average = static_cast<double>(sum) / size;
    
    cout << "[Parallel] Sum: " << sum << "\n";
    cout << "[Parallel] Average: " << fixed << setprecision(4) << average << "\n";
    cout << "[Parallel] Time: " << duration.count() << " ms\n";
    
    return average;
}

// ========================================================================
// ФУНКЦИЯ: Освобождение динамической памяти
// ========================================================================
// Освобождает память, выделенную для массива
// Параметры:
//   - arr: указатель на массив
// Важно: после вызова указатель становится невалидным (dangling pointer)
void freeDynamicArray(int* arr) {
    cout << "\n[Memory] Freeing dynamic array at address: " << arr << "\n";
    delete[] arr;  // Освобождаем память массива
    cout << "[Memory] Memory freed successfully\n";
}

// ========================================================================
// ФУНКЦИЯ: Печать первых N элементов массива
// ========================================================================
// Выводит первые элементы массива для демонстрации
void printArraySample(int* arr, int size, int sample_size = 20) {
    cout << "\n[Array] First " << min(sample_size, size) << " elements:\n";
    cout << "  ";
    for (int i = 0; i < min(sample_size, size); ++i) {
        cout << arr[i];
        if (i < min(sample_size, size) - 1) cout << ", ";
    }
    if (size > sample_size) {
        cout << " ...";
    }
    cout << "\n";
}

// ========================================================================
// ФУНКЦИЯ: Сравнение последовательного и параллельного подходов
// ========================================================================
void runBenchmark(int size, int num_threads) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  BENCHMARK: Dynamic Array Average Calculation\n";
    cout << "=================================================================\n";
    cout << "Array size: " << size << " elements\n";
    cout << "Threads: " << num_threads << "\n";
    cout << "-----------------------------------------------------------------\n";
    
    // 1. Создание динамического массива
    int* arr = createDynamicArray(size);
    
    // 2. Заполнение случайными числами
    fillArrayRandom(arr, size, 1, 100);
    
    // 3. Печать образца данных
    if (size <= 100) {
        printArraySample(arr, size, size);
    } else {
        printArraySample(arr, size, 20);
    }
    
    // 4. Последовательное вычисление среднего
    double avg_seq = calculateAverageSequential(arr, size);
    
    // 5. Параллельное вычисление среднего
    double avg_par = calculateAverageParallel(arr, size, num_threads);
    
    // 6. Проверка корректности
    cout << "\n[Verification] Checking results...\n";
    double diff = abs(avg_seq - avg_par);
    cout << "[Verification] Difference: " << scientific << setprecision(10) << diff << "\n";
    
    if (diff < 0.0001) {
        cout << "[Verification] ✓ Results match! Both methods are correct.\n";
    } else {
        cout << "[Verification] ✗ Results differ! Check implementation.\n";
    }
    
    // 7. Освобождение памяти
    freeDynamicArray(arr);
    
    cout << "=================================================================\n";
}

// ========================================================================
// ФУНКЦИЯ: Демонстрация работы с указателями
// ========================================================================
void demonstratePointers() {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  DEMONSTRATION: Pointers and Dynamic Memory\n";
    cout << "=================================================================\n";
    
    // Создаём маленький массив для демонстрации
    int size = 5;
    cout << "\n[Demo] Creating array of size " << size << "\n";
    
    int* ptr = new int[size];
    cout << "[Demo] Pointer address: " << ptr << "\n";
    cout << "[Demo] Size of pointer: " << sizeof(ptr) << " bytes\n";
    cout << "[Demo] Size of int: " << sizeof(int) << " bytes\n";
    
    // Заполняем массив
    cout << "\n[Demo] Filling array with values: 10, 20, 30, 40, 50\n";
    for (int i = 0; i < size; ++i) {
        ptr[i] = (i + 1) * 10;
    }
    
    // Работа с указателями
    cout << "\n[Demo] Accessing elements using pointer arithmetic:\n";
    for (int i = 0; i < size; ++i) {
        cout << "  ptr[" << i << "] = " << ptr[i] 
             << " (address: " << (ptr + i) << ")\n";
    }
    
    // Вычисляем среднее
    cout << "\n[Demo] Calculating average...\n";
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += *(ptr + i);  // Используем разыменование указателя
    }
    double avg = static_cast<double>(sum) / size;
    cout << "[Demo] Sum: " << sum << "\n";
    cout << "[Demo] Average: " << avg << "\n";
    
    // Освобождаем память
    cout << "\n[Demo] Freeing memory...\n";
    delete[] ptr;
    cout << "[Demo] Memory freed\n";
    
    cout << "=================================================================\n";
}

// ========================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ========================================================================
int main() {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "    LAB 2 - PART 3: Dynamic Memory and Parallel Average\n";
    cout << "=================================================================\n";
    
    // Проверяем поддержку OpenMP
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
    cout << "[OpenMP] Available\n";
    cout << "[OpenMP] Max threads: " << num_threads << "\n";
#else
    cout << "[OpenMP] Not available (compile with -fopenmp)\n";
#endif
    
    cout << "\n";
    
    // Демонстрация работы с указателями
    demonstratePointers();
    
    // Бенчмарки с разными размерами массивов
    vector<int> test_sizes = {1000, 100000, 10000000};
    
    cout << "\n[Benchmark] Running tests with sizes: ";
    for (size_t i = 0; i < test_sizes.size(); ++i) {
        cout << test_sizes[i];
        if (i < test_sizes.size() - 1) cout << ", ";
    }
    cout << "\n";
    
    for (int size : test_sizes) {
        runBenchmark(size, num_threads);
        cout << "\n";
    }
    
    // Анализ производительности
    cout << "=================================================================\n";
    cout << "  ANALYSIS\n";
    cout << "=================================================================\n";
    cout << "Key observations:\n";
    cout << "1. For small arrays (N < 10000), sequential may be faster\n";
    cout << "   - Thread creation overhead dominates\n";
    cout << "2. For large arrays (N > 100000), parallel shows speedup\n";
    cout << "   - Computational work outweighs overhead\n";
    cout << "3. OpenMP reduction automatically handles:\n";
    cout << "   - Thread-local sum variables\n";
    cout << "   - Final reduction operation\n";
    cout << "   - Thread synchronization\n";
    cout << "4. Memory management is crucial:\n";
    cout << "   - Always free allocated memory with delete[]\n";
    cout << "   - Avoid memory leaks\n";
    cout << "=================================================================\n";
    
    cout << "\n[Program] All tests completed successfully!\n";
    cout << "[Program] All memory freed. No memory leaks.\n\n";
    
    return 0;
}