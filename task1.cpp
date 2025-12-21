#include <iostream>
#include <random>
#include <cstdint>

int main() {
    const int N = 50000;

    // 1) выделяю памяти
    int* a = new int[N];

    // 2) заполняю случайными числами
    std::mt19937_64 gen(123456789ULL);
    std::uniform_int_distribution<int> dist(1, 100);
    for (int i = 0; i < N; ++i) a[i] = dist(gen);

    // 3) среднее подсчет
    long long sum = 0;
    for (int i = 0; i < N; ++i) sum += a[i];
    double avg = static_cast<double>(sum) / N;

    std::cout << "N=" << N << ", average=" << avg << "\n";

    // 4) освобождаю память
    delete[] a;
    return 0;
}