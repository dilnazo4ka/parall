#include <bits/stdc++.h>              // подключаю стандартные C++ библиотеки
#include <mpi.h>                      // подключаю библиотеку MPI
using namespace std;                  // использую пространство имён std

int main(int argc, char** argv) {     // точка входа MPI-программы
  MPI_Init(&argc, &argv);             // инициализирую MPI окружение

  int rank = 0;                       // номер текущего процесса
  int size = 1;                       // общее число процессов
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // получаю rank процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size); // получаю количество процессов

  const int N = 1'000'000;            // общий размер задачи
  int base = N / size;                // базовое число элементов на процесс
  int rem  = N % size;                // остаток от деления
  int local_n = base + (rank < rem);  // размер локального массива

  vector<double> local(local_n);      // локальный массив процесса

  for (int i = 0; i < local_n; i++)   // инициализирую локальные данные
    local[i] = 1.0;                   // заполняю единицами

  MPI_Barrier(MPI_COMM_WORLD);        // синхронизирую процессы перед замером

  double t1 = MPI_Wtime();            // фиксирую время начала вычислений

  double local_sum = 0.0;             // локальная сумма
  for (int i = 0; i < local_n; i++)   // считаю сумму локального массива
    local_sum += local[i];            // накапливаю сумму

  double global_sum = 0.0;            // переменная для глобальной суммы
  MPI_Reduce(&local_sum,              // отправляю локальную сумму
             &global_sum,             // принимаю результат на root
             1,                       // количество элементов
             MPI_DOUBLE,              // тип данных
             MPI_SUM,                 // операция суммирования
             0,                       // root процесс
             MPI_COMM_WORLD);         // коммуникатор

  double t2 = MPI_Wtime();            // фиксирую время окончания

  if (rank == 0) {                    // только root выводит результат
    cout.setf(std::ios::fixed);       // фиксированный формат вывода
    cout << setprecision(6);
    cout << "N=" << N << "\n";         // размер задачи
    cout << "processes=" << size << "\n"; // число процессов
    cout << "time_s=" << (t2 - t1) << "\n"; // время выполнения
    cout << "sum=" << global_sum << "\n";  // итоговая сумма
  }

  MPI_Finalize();                     // завершаю MPI окружение
  return 0;                           // корректный выход
}
