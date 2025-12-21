#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

using namespace std;
using namespace chrono;
// Функция для создания и заполнения массива случайными числами
static vector<int> make_array(int n) { 
    mt19937_64 gen(123456789ULL); // Фиксированный сид для воспроизводимости
    uniform_int_distribution<int> dist(1, 1000000);
    vector<int> a(n);
    for (int &x : a) x = dist(gen);
    return a;
}

int main() {
    const int N = 1'000'000;
    auto a = make_array(N);
    // Последовательный поиск мин и макс
    int mn = numeric_limits<int>::max();
    int mx = numeric_limits<int>::min();
    // Засекаю время
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
    }
    auto t1 = high_resolution_clock::now();
    // Вычисляю время в миллисекундах
    double ms = duration<double, milli>(t1 - t0).count();
    cout << "seq min=" << mn << " max=" << mx << " time=" << ms << " ms\n";
    return 0;
}