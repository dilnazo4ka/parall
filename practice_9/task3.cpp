#include <bits/stdc++.h>              // подключаю стандартные C++ библиотеки
#include <mpi.h>                      // подключаю библиотеку MPI
using namespace std;                  // использую пространство имён std

const double INF = 1e12;              // большое число, используем как бесконечность

int owner_of_row(int k,               // функция определяет, какому процессу принадлежит строка k
                 const vector<int>& start, // массив начальных индексов строк процессов
                 const vector<int>& cnt){  // массив количества строк у процессов
  for(int p = 0; p < (int)start.size(); p++){ // перебираю все процессы
    if(k >= start[p] && k < start[p] + cnt[p]) // проверяю, попадает ли строка k в диапазон процесса p
      return p;                          // возвращаю номер процесса-владельца
  }
  return 0;                              // запасной вариант (не должен срабатывать)
}

int main(int argc, char** argv){       // точка входа MPI-программы
  MPI_Init(&argc, &argv);               // инициализирую MPI

  int rank, size;                       // переменные для номера процесса и их количества
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // получаю номер текущего процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size); // получаю общее число процессов

  int N = 128;                          // размер графа по умолчанию
  if(argc > 1)                          // если передан аргумент командной строки
    N = atoi(argv[1]);                 // читаю размер графа из аргумента

  vector<int> rows(size), start(size); // массивы количества строк и стартовых индексов
  int base = N / size;                 // базовое число строк на процесс
  int rem = N % size;                  // остаток строк

  for(int p = 0; p < size; p++)        // распределяю строки между процессами
    rows[p] = base + (p < rem ? 1 : 0); // первым rem процессам даю на одну строку больше

  start[0] = 0;                        // первая строка начинается с нуля
  for(int p = 1; p < size; p++)        // считаю стартовый индекс строк для каждого процесса
    start[p] = start[p-1] + rows[p-1]; // накапливаю количество строк

  int local_rows = rows[rank];         // число строк у текущего процесса

  vector<double> localD(local_rows * N); // локальный блок матрицы расстояний
  vector<double> D;                    // полная матрица (используется только на rank 0)

  if(rank == 0){                       // инициализацию делает только нулевой процесс
    D.resize(N * N);                   // выделяю память под всю матрицу
    mt19937 gen(42);                   // генератор случайных чисел
    uniform_real_distribution<double> dist(1.0, 20.0); // диапазон весов рёбер

    for(int i = 0; i < N; i++){        // цикл по строкам
      for(int j = 0; j < N; j++){      // цикл по столбцам
        if(i == j)                     // если диагональный элемент
          D[i*N + j] = 0.0;            // расстояние до себя равно нулю
        else if(gen() % 4 == 0)        // случайно убираю рёбра
          D[i*N + j] = INF;            // если ребра нет — бесконечность
        else
          D[i*N + j] = dist(gen);      // иначе случайный вес ребра
      }
    }
  }

  vector<int> sc(size), dsp(size);     // массивы для Scatterv/Gatherv
  for(int p = 0; p < size; p++)        // считаю количество элементов для каждого процесса
    sc[p] = rows[p] * N;               // строки процесса * N столбцов

  dsp[0] = 0;                          // смещение для первого процесса
  for(int p = 1; p < size; p++)        // считаю смещения
    dsp[p] = dsp[p-1] + sc[p-1];       // накопление смещений

  MPI_Scatterv(                        // распределяю строки матрицы по процессам
    rank==0 ? D.data() : nullptr,      // источник данных (только у rank 0)
    sc.data(),                          // количество элементов для каждого процесса
    dsp.data(),                         // смещения
    MPI_DOUBLE,                        // тип данных
    localD.data(),                     // локальный буфер
    local_rows * N,                    // размер локального блока
    MPI_DOUBLE,                        // тип данных
    0,                                 // root процесс
    MPI_COMM_WORLD);                   // коммуникатор

  vector<double> kth_row(N);           // буфер для строки k

  double t1 = MPI_Wtime();             // начало измерения времени

  for(int k = 0; k < N; k++){          // основной цикл алгоритма Флойда–Уоршелла
    int owner = owner_of_row(k, start, rows); // определяю владельца строки k

    if(rank == owner){                 // если текущий процесс владеет строкой k
      int loc = k - start[rank];       // локальный индекс строки k
      for(int j = 0; j < N; j++)       // копирую строку k в буфер
        kth_row[j] = localD[loc*N + j];
    }

    MPI_Bcast(                         // рассылаю строку k всем процессам
      kth_row.data(),                  // буфер строки
      N,                               // длина строки
      MPI_DOUBLE,                      // тип данных
      owner,                           // процесс-источник
      MPI_COMM_WORLD);                 // коммуникатор

    for(int i = 0; i < local_rows; i++){ // обновляю локальные строки
      for(int j = 0; j < N; j++){        // перебираю столбцы
        double via_k = localD[i*N + k] + kth_row[j]; // путь через вершину k
        if(via_k < localD[i*N + j])     // если путь короче
          localD[i*N + j] = via_k;      // обновляю расстояние
      }
    }
  }

  double t2 = MPI_Wtime();             // конец измерения времени

  MPI_Gatherv(                         // собираю матрицу обратно на rank 0
    localD.data(),                     // локальный буфер
    local_rows * N,                    // размер локальных данных
    MPI_DOUBLE,                        // тип данных
    rank==0 ? D.data() : nullptr,      // приёмный буфер (только у rank 0)
    sc.data(),                         // размеры блоков
    dsp.data(),                        // смещения
    MPI_DOUBLE,                        // тип данных
    0,                                 // root процесс
    MPI_COMM_WORLD);                   // коммуникатор

  if(rank == 0){                       // вывод делает только нулевой процесс
    cout.setf(std::ios::fixed);        // фиксированный формат вывода
    cout << setprecision(6);           // точность вывода
    cout << "N=" << N << "\n";          // размер графа
    cout << "Floyd-Warshall MPI time = " << (t2 - t1) << " s\n"; // время выполнения
    cout << "check D[0][1] = " << D[1] << "\n"; // быстрая проверка результата
  }

  MPI_Finalize();                      // завершаю MPI окружение
  return 0;                            // корректный выход из программы
}