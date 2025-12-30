#include <iostream>     // ввод и вывод
#include <vector>       // работа с массивами
#include <random>       // генерация случайных чисел
#include <chrono>       // измерение времени
#include <omp.h>        // OpenMP

// Последовательная сортировка выбором
void selection_sort_seq(std::vector<int>& a) {
    int n = a.size();

    // внешний цикл — фиксирую позицию минимального элемента
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;

        // внутренний цикл — ищу минимальный элемент
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < a[min_idx])
                min_idx = j;
        }

        // меняю местами найденный минимум и текущий элемент
        std::swap(a[i], a[min_idx]);
    }
}

// Параллельная версия сортировки выбором
void selection_sort_omp(std::vector<int>& a) {
    int n = a.size();

    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;

        // распараллеливаю поиск минимума
        #pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            // использую critical, чтобы избежать гонки данных
            #pragma omp critical
            {
                if (a[j] < a[min_idx])
                    min_idx = j;
            }
        }

        // выполняю обмен элементов
        std::swap(a[i], a[min_idx]);
    }
}

int main() {
    // проверяю три размера массива
    std::vector<int> sizes = {10000, 100000, 400000};

    for (int n : sizes) {
        // создаю базовый массив
        std::vector<int> base(n);

        // инициализирую генератор случайных чисел
        std::mt19937 gen(123);
        std::uniform_int_distribution<int> dist(1, 100000);

        // заполняю массив
        for (int &x : base)
            x = dist(gen);

        /* ---------- Последовательная версия ---------- */
        auto a = base;
        auto t1 = std::chrono::high_resolution_clock::now();
        selection_sort_seq(a);
        auto t2 = std::chrono::high_resolution_clock::now();
        double seq_time =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        /* ---------- Параллельная версия ---------- */
        a = base;
        t1 = std::chrono::high_resolution_clock::now();
        selection_sort_omp(a);
        t2 = std::chrono::high_resolution_clock::now();
        double par_time =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        // вывожу результаты
        std::cout << "N=" << n
                  << " | seq=" << seq_time << " ms"
                  << " | omp=" << par_time << " ms\n";
    }
}
