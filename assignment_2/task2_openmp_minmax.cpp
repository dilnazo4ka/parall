#include <iostream>     // подключаю библиотеку для ввода/вывода
#include <vector>       // подключаю библиотеку для работы с динамическими массивами
#include <random>       // подключаю генератор случайных чисел
#include <chrono>       // подключаю библиотеку для измерения времени
#include <omp.h>        // подключаю OpenMP для параллельных вычислений

int main() {
    // задаю размер массива
    const int N = 10000;

    // создаю вектор (массив) из N элементов
    std::vector<int> a(N);

    // инициализирую генератор случайных чисел
    std::mt19937 gen(42);

    // задаю диапазон случайных чисел
    std::uniform_int_distribution<int> dist(1, 100000);

    // заполняю массив случайными значениями
    for (int &x : a)
        x = dist(gen);

    /* ================= Последовательная версия ================= */

    // фиксирую время начала последовательного алгоритма
    auto t1 = std::chrono::high_resolution_clock::now();

    // инициализирую минимальное и максимальное значения первым элементом
    int min_seq = a[0];
    int max_seq = a[0];

    // прохожу по массиву последовательно
    for (int i = 1; i < N; ++i) {
        // сравниваю текущий элемент с минимумом
        if (a[i] < min_seq)
            min_seq = a[i];
        // сравниваю текущий элемент с максимумом
        if (a[i] > max_seq)
            max_seq = a[i];
    }

    // фиксирую время окончания
    auto t2 = std::chrono::high_resolution_clock::now();

    // вычисляю время выполнения в миллисекундах
    double seq_time =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    /* ================= Параллельная версия (OpenMP) ================= */

    // инициализирую переменные для параллельного поиска
    int min_par = a[0];
    int max_par = a[0];

    // снова фиксирую время начала
    t1 = std::chrono::high_resolution_clock::now();

    // запускаю параллельный цикл OpenMP
    // использую reduction, чтобы корректно найти min и max
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par)
    for (int i = 0; i < N; ++i) {
        if (a[i] < min_par)
            min_par = a[i];
        if (a[i] > max_par)
            max_par = a[i];
    }

    // фиксирую время окончания
    t2 = std::chrono::high_resolution_clock::now();

    // считаю время параллельного выполнения
    double par_time =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    // вывожу результаты последовательной версии
    std::cout << "Sequential: min=" << min_seq
              << " max=" << max_seq
              << " time=" << seq_time << " ms\n";

    // вывожу результаты параллельной версии
    std::cout << "Parallel:   min=" << min_par
              << " max=" << max_par
              << " time=" << par_time << " ms\n";
}
