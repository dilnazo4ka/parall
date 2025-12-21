#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h>

using namespace std;
using namespace chrono;

static vector<int> make_array(int n) { // Функция для создания и заполнения массива случайными числами как в task2.cpp
    mt19937_64 gen(123456789ULL);
    uniform_int_distribution<int> dist(1, 1000000);
    vector<int> a(n);
    for (int &x : a) x = dist(gen);
    return a;
}

int main() {
    const int N = 1'000'000;
    auto a = make_array(N);
    // Последовательный поиск мин и макс
    int mn_seq = numeric_limits<int>::max();
    int mx_seq = numeric_limits<int>::min();

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        if (a[i] < mn_seq) mn_seq = a[i];
        if (a[i] > mx_seq) mx_seq = a[i];
    }
    auto t1 = high_resolution_clock::now();
    double ms_seq = duration<double, milli>(t1 - t0).count();
    // Параллельный поиск мин и макс с OpenMP
    int mn = numeric_limits<int>::max();
    int mx = numeric_limits<int>::min();

    auto t2 = high_resolution_clock::now();

    #pragma omp parallel // Начало параллельной области
    {
        int local_min = numeric_limits<int>::max();
        int local_max = numeric_limits<int>::min();

        #pragma omp for nowait schedule(static) // Распределение итераций цикла между потоками
        for (int i = 0; i < N; ++i) {
            if (a[i] < local_min) local_min = a[i];
            if (a[i] > local_max) local_max = a[i];
        }

        #pragma omp critical // Критическая секция для обновления глобальных мин и макс
        {
            if (local_min < mn) mn = local_min;
            if (local_max > mx) mx = local_max;
        }
    }

    auto t3 = high_resolution_clock::now();
    double ms_omp = duration<double, milli>(t3 - t2).count();

    cout << "SEQ min=" << mn_seq << " max=" << mx_seq << " time=" << ms_seq << " ms\n";
    cout << "OMP min=" << mn     << " max=" << mx     << " time=" << ms_omp << " ms\n";
    cout << "Speedup: " << (ms_seq / max(1e-12, ms_omp)) << "\n";

    return 0;
}