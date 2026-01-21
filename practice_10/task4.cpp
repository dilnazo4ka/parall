#include <bits/stdc++.h>              // подключаю стандартные C++ библиотеки
#include <mpi.h>                      // подключаю MPI
using namespace std;                  // использую пространство имён std

int main(int argc, char** argv){      // точка входа в MPI-программу
  MPI_Init(&argc, &argv);             // инициализирую MPI окружение

  int rank = 0;                       // номер текущего процесса
  int size = 1;                       // общее число процессов
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // получаю rank процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size); // получаю количество процессов

  const int N = 10'000'000;           // размер массива (фиксированный для strong scaling)
  vector<double> local;               // локальный кусок массива

  int base = N / size;                // базовое число элементов на процесс
  int rem  = N % size;                // остаток от деления
  int local_n = base + (rank < rem ? 1 : 0); // размер локального массива

  local.resize(local_n);              // выделяю память под локальный массив

  for(int i = 0; i < local_n; i++)    // инициализирую локальные данные
    local[i] = double(rank + 1);      // заполняю значениями (зависят от rank)

  MPI_Barrier(MPI_COMM_WORLD);        // синхронизирую все процессы перед замером

  double t1 = MPI_Wtime();             // фиксирую время начала вычислений

  double local_sum = 0.0;              // локальная сумма
  for(int i = 0; i < local_n; i++)     // считаю сумму элементов локального массива
    local_sum += local[i];             // накапливаю сумму

  double global_sum = 0.0;             // глобальная сумма
  MPI_Reduce(&local_sum,               // отправляю локальную сумму
             &global_sum,              // принимаю глобальную сумму на root
             1,                        // один элемент
             MPI_DOUBLE,               // тип данных
             MPI_SUM,                  // операция суммирования
             0,                        // root процесс
             MPI_COMM_WORLD);          // коммуникатор

  double t2 = MPI_Wtime();             // фиксирую время окончания вычислений

  if(rank == 0){                       // только root процесс выводит результат
    cout.setf(std::ios::fixed);        // фиксированный формат вывода
    cout << setprecision(6);           // точность вывода
    cout << "N=" << N << "\n";          // размер задачи
    cout << "processes=" << size << "\n"; // количество MPI процессов
    cout << "time=" << (t2 - t1) << " s\n"; // время выполнения
    cout << "sum=" << global_sum << "\n";  // итоговая сумма
  }

  MPI_Finalize();                      // завершаю MPI окружение
  return 0;                            // корректный выход из программы
}
