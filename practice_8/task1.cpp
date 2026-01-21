#include <bits/stdc++.h>          // стандартные C++ библиотеки
#include <omp.h>                  // OpenMP
using namespace std;

int main() {
  const int N = 1'000'000;        // размер массива
  vector<float> a(N);             // массив данных

  // инициализация массива
  for (int i = 0; i < N; i++)
    a[i] = float(i);

  // старт замера времени
  double t1 = omp_get_wtime();    // время до вычислений

  #pragma omp parallel for        // распараллеливание цикла
  for (int i = 0; i < N; i++) {
    a[i] = a[i] * 2.0f;           // обработка: умножение на 2
  }

  double t2 = omp_get_wtime();    // время после вычислений

  // вывод результатов
  cout.setf(std::ios::fixed);
  cout << setprecision(6);
  cout << "N=" << N << "\n";
  cout << "CPU(OpenMP) time = " << (t2 - t1) * 1000 << " ms\n";
  cout << "check a[100]=" << a[100] << "\n"; // проверка корректности

  return 0;
}
