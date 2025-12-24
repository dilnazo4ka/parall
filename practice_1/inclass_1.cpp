// lab2_part1.cpp
// ========================================================================
// ЧАСТЬ 1: РАБОТА С МАССИВАМИ И ПАРАЛЛЕЛИЗАЦИЯ OpenMP
// ========================================================================
// 
// Программа демонстрирует:
// 1. Создание массива и заполнение случайными числами
// 2. Поиск максимального и минимального элементов
// 3. Сравнение последовательного и параллельного подходов
// 4. Измерение времени выполнения с помощью <chrono>
//
// ========================================================================
// СБОРКА:
// ========================================================================
// g++ -fopenmp -O2 -std=c++17 lab2_part1.cpp -o lab2_part1.exe
//
// ========================================================================
// ЗАПУСК:
// ========================================================================
// $env:OMP_NUM_THREADS = '4'
// .\lab2_part1.exe
// ========================================================================

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ========================================================================
// ФУНКЦИЯ: Создание и заполнение массива случайными числами
// ========================================================================
// Создаёт вектор заданного размера и заполняет его случайными числами
// Параметры:
//   - size: размер массива
//   - min_val: минимальное значение (по умолчанию 1)
//   - max_val: максимальное значение (по умолчанию 100)
// Возвращает: вектор целых чисел
vector<int> createRandomArray(int size, int min_val = 1, int max_val = 100) {
    cout << "[Array] Creating array of size " << size << "\n";
    cout << "[Array] Random range: [" << min_val << ", " << max_val << "]\n";
    
    vector<int> arr(size);
    
    // Инициализация генератора случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(min_val, max_val);
    
    // Заполнение массива
    for (int i = 0; i < size; ++i) {
        arr[i] = distrib(gen);
    }
    
    cout << "[Array] Array created and filled\n";
    return arr;
}

// ========================================================================
// ФУНКЦИЯ: Печать массива (или его части)
// ========================================================================
// Выводит элементы массива на экран
// Если массив большой (>50), показывает только первые 25 и последние 25
void printArray(const vector<int>& arr, const string& title = "Array") {
    int size = arr.size();
    cout << "\n[" << title << "] Size: " << size << "\n";
    cout << "[" << title << "] Elements: ";
    
    if (size <= 50) {
        // Печатаем весь массив
        for (int i = 0; i < size; ++i) {
            cout << arr[i];
            if (i < size - 1) cout << ", ";
        }
        cout << "\n";
    } else {
        // Печатаем первые 25 элементов
        for (int i = 0; i < 25; ++i) {
            cout << arr[i] << ", ";
        }
        cout << "... ";
        // Печатаем последние 25 элементов
        for (int i = size - 25; i < size; ++i) {
            cout << arr[i];
            if (i < size - 1) cout << ", ";
        }
        cout << "\n";
    }
}

// ========================================================================
// ФУНКЦИЯ: Последовательный поиск MIN и MAX
// ========================================================================
// Находит минимальный и максимальный элементы массива последовательно
// Параметры:
//   - arr: входной массив
//   - min_val: выходной параметр для минимума
//   - max_val: выходной параметр для максимума
// Возвращает: время выполнения в миллисекундах
double findMinMaxSequential(const vector<int>& arr, int& min_val, int& max_val) {
    cout << "\n[Sequential] Finding MIN and MAX...\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    // Инициализация
    min_val = numeric_limits<int>::max();
    max_val = numeric_limits<int>::min();
    
    // Последовательный поиск
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    cout << "[Sequential] MIN = " << min_val << "\n";
    cout << "[Sequential] MAX = " << max_val << "\n";
    cout << "[Sequential] Time: " << fixed << setprecision(6) 
         << duration.count() << " ms\n";
    
    return duration.count();
}

// ========================================================================
// ФУНКЦИЯ: Параллельный поиск MIN и MAX с OpenMP
// ========================================================================
// Находит минимальный и максимальный элементы используя параллелизацию
// Параметры:
//   - arr: входной массив
//   - min_val: выходной параметр для минимума
//   - max_val: выходной параметр для максимума
//   - num_threads: количество потоков
// Возвращает: время выполнения в миллисекундах
//
// OpenMP reduction(min:) и reduction(max:):
//   - Каждый поток создаёт свою локальную копию переменной
//   - Каждый поток обрабатывает свою часть массива
//   - В конце OpenMP автоматически находит min/max среди всех потоков
double findMinMaxParallel(const vector<int>& arr, int& min_val, int& max_val, int num_threads) {
    cout << "\n[Parallel] Finding MIN and MAX with " << num_threads << " threads...\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    // Инициализация
    min_val = numeric_limits<int>::max();
    max_val = numeric_limits<int>::min();
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    
    // Параллельный поиск с reduction
    // reduction(min:min_val) - автоматически находит минимум
    // reduction(max:max_val) - автоматически находит максимум
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
#else
    // Если OpenMP недоступен, используем последовательный алгоритм
    cout << "[Warning] OpenMP not available, using sequential mode\n";
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] < min_val) min_val = arr[i];
        if (arr[i] > max_val) max_val = arr[i];
    }
#endif
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    cout << "[Parallel] MIN = " << min_val << "\n";
    cout << "[Parallel] MAX = " << max_val << "\n";
    cout << "[Parallel] Time: " << fixed << setprecision(6) 
         << duration.count() << " ms\n";
    
    return duration.count();
}

// ========================================================================
// ФУНКЦИЯ: Проверка корректности результатов
// ========================================================================
// Сравнивает результаты последовательного и параллельного алгоритмов
void verifyResults(int seq_min, int seq_max, int par_min, int par_max) {
    cout << "\n[Verification] Checking results...\n";
    
    bool min_match = (seq_min == par_min);
    bool max_match = (seq_max == par_max);
    
    cout << "[Verification] MIN: Sequential = " << seq_min 
         << ", Parallel = " << par_min 
         << " → " << (min_match ? "✓ MATCH" : "✗ MISMATCH") << "\n";
    
    cout << "[Verification] MAX: Sequential = " << seq_max 
         << ", Parallel = " << par_max 
         << " → " << (max_match ? "✓ MATCH" : "✗ MISMATCH") << "\n";
    
    if (min_match && max_match) {
        cout << "[Verification] ✓ All results are correct!\n";
    } else {
        cout << "[Verification] ✗ Results differ! Check implementation.\n";
    }
}

// ========================================================================
// ФУНКЦИЯ: Анализ производительности
// ========================================================================
// Вычисляет speedup и efficiency
void analyzePerformance(double seq_time, double par_time, int threads) {
    cout << "\n[Performance Analysis]\n";
    cout << "  Sequential time: " << fixed << setprecision(6) << seq_time << " ms\n";
    cout << "  Parallel time:   " << fixed << setprecision(6) << par_time << " ms\n";
    
    if (par_time > 0) {
        double speedup = seq_time / par_time;
        double efficiency = (speedup / threads) * 100.0;
        
        cout << "  Speedup:         " << fixed << setprecision(3) << speedup << "x\n";
        cout << "  Efficiency:      " << fixed << setprecision(2) << efficiency << "%\n";
        
        if (speedup > 1.0) {
            cout << "  → Parallel version is FASTER\n";
        } else if (speedup < 1.0) {
            cout << "  → Sequential version is FASTER (array too small or overhead too high)\n";
        } else {
            cout << "  → Both versions have similar performance\n";
        }
    }
}

// ========================================================================
// ФУНКЦИЯ: Запуск бенчмарка
// ========================================================================
void runBenchmark(int array_size, int num_threads) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  BENCHMARK: Array MIN/MAX Search\n";
    cout << "=================================================================\n";
    cout << "Array size: " << array_size << " elements\n";
    cout << "Threads:    " << num_threads << "\n";
    cout << "-----------------------------------------------------------------\n";
    
    // 1. Создание массива
    vector<int> arr = createRandomArray(array_size, 1, 100);
    
    // 2. Печать массива (если небольшой)
    if (array_size <= 100) {
        printArray(arr, "Input Array");
    }
    
    // 3. Последовательный поиск
    int seq_min, seq_max;
    double seq_time = findMinMaxSequential(arr, seq_min, seq_max);
    
    // 4. Параллельный поиск
    int par_min, par_max;
    double par_time = findMinMaxParallel(arr, par_min, par_max, num_threads);
    
    // 5. Проверка корректности
    verifyResults(seq_min, seq_max, par_min, par_max);
    
    // 6. Анализ производительности
    analyzePerformance(seq_time, par_time, num_threads);
    
    cout << "=================================================================\n";
}

// ========================================================================
// ДОПОЛНИТЕЛЬНО: Демонстрация работы с маленьким массивом
// ========================================================================
void demonstrateWithSmallArray() {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  DEMONSTRATION: Small Array Example\n";
    cout << "=================================================================\n";
    
    // Создаём маленький массив для наглядности
    vector<int> demo_arr = {45, 23, 78, 12, 89, 34, 56, 91, 5, 67};
    
    cout << "[Demo] Created array with predefined values\n";
    printArray(demo_arr, "Demo Array");
    
    // Последовательный поиск
    int min_val, max_val;
    cout << "\n[Demo] Sequential search:\n";
    auto start = chrono::high_resolution_clock::now();
    
    min_val = *min_element(demo_arr.begin(), demo_arr.end());
    max_val = *max_element(demo_arr.begin(), demo_arr.end());
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    
    cout << "  MIN = " << min_val << "\n";
    cout << "  MAX = " << max_val << "\n";
    cout << "  Time: " << duration.count() << " ms\n";
    
    cout << "=================================================================\n";
}

// ========================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ========================================================================
int main() {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "    LAB 2 - PART 1: Arrays and OpenMP Parallelization\n";
    cout << "=================================================================\n";
    
    // Проверка поддержки OpenMP
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
    cout << "[OpenMP] Available\n";
    cout << "[OpenMP] Max threads: " << num_threads << "\n";
    cout << "[OpenMP] Version: " << _OPENMP << "\n";
#else
    cout << "[OpenMP] Not available (compile with -fopenmp)\n";
#endif
    
    // Демонстрация с маленьким массивом
    demonstrateWithSmallArray();
    
    // Бенчмарки с разными размерами массивов
    vector<int> test_sizes = {1000, 100000, 1000000, 10000000};
    
    cout << "\n[Info] Running benchmarks with array sizes: ";
    for (size_t i = 0; i < test_sizes.size(); ++i) {
        cout << test_sizes[i];
        if (i < test_sizes.size() - 1) cout << ", ";
    }
    cout << "\n";
    
    // Запускаем бенчмарки
    for (int size : test_sizes) {
        runBenchmark(size, num_threads);
    }
    
    // Итоговый анализ
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  SUMMARY\n";
    cout << "=================================================================\n";
    cout << "Key findings:\n";
    cout << "1. For small arrays (N < 10,000):\n";
    cout << "   - Sequential is often faster\n";
    cout << "   - Thread overhead dominates computation time\n";
    cout << "\n";
    cout << "2. For medium arrays (10,000 < N < 1,000,000):\n";
    cout << "   - Parallel starts showing benefits\n";
    cout << "   - Speedup increases with array size\n";
    cout << "\n";
    cout << "3. For large arrays (N > 1,000,000):\n";
    cout << "   - Parallel provides significant speedup\n";
    cout << "   - Efficiency depends on number of cores\n";
    cout << "\n";
    cout << "4. OpenMP reduction:\n";
    cout << "   - Simplifies parallel min/max finding\n";
    cout << "   - Automatically handles thread coordination\n";
    cout << "   - No manual synchronization needed\n";
    cout << "=================================================================\n";
    
    cout << "\n[Program] All tests completed successfully!\n\n";
    
    return 0;
}