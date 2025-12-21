#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

static vector<int> make_array(int n) { // Функция для создания и заполнения массива случайными числами
    mt19937_64 gen(123456789ULL);
    uniform_int_distribution<int> dist(1, 100);
    vector<int> a(n);
    for (int &x : a) x = dist(gen);
    return a;
}

int main() {
    const int N = 5'000'000;
    auto a = make_array(N);

    auto t0 = high_resolution_clock::now();
    long long sum_seq = 0;
    for (int i = 0; i < N; ++i) sum_seq += a[i]; // Последовательный подсчет суммы
    double avg_seq = (double)sum_seq / N;
    auto t1 = high_resolution_clock::now();
    double ms_seq = duration<double, milli>(t1 - t0).count();

    auto t2 = high_resolution_clock::now();
    long long sum_omp = 0;

    #pragma omp parallel for reduction(+:sum_omp) schedule(static) // Параллельный подсчет суммы с OpenMP reduction
    for (int i = 0; i < N; ++i) {
        sum_omp += a[i];
    }

    double avg_omp = (double)sum_omp / N; // Вычисление среднего
    auto t3 = high_resolution_clock::now();
    double ms_omp = duration<double, milli>(t3 - t2).count();

    cout << "SEQ avg=" << avg_seq << " time=" << ms_seq << " ms\n"; // Вывод результатов
    cout << "OMP avg=" << avg_omp << " time=" << ms_omp << " ms\n";
    cout << "Speedup: " << (ms_seq / max(1e-12, ms_omp)) << "\n"; // Вывод ускорения

    return 0;
}