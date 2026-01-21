#include <bits/stdc++.h>                    // стандартные C++ библиотеки
#include <mpi.h>                            // MPI
using namespace std;                        // std::

int main(int argc, char** argv){            // вход в программу
  MPI_Init(&argc, &argv);                   // инициализирую MPI

  int rank = 0;                             // ранг процесса
  int size = 1;                             // число процессов
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // получаю rank
  MPI_Comm_size(MPI_COMM_WORLD, &size);     // получаю size

  long long N = 1000000;                    // размер массива по умолчанию
  if(argc >= 2) N = atoll(argv[1]);         // если передали N аргументом

  vector<double> data;                      // полный массив (только у rank 0)
  if(rank == 0){                            // только главный процесс
    data.resize(N);                         // выделяю память под массив
    mt19937_64 gen(42);                     // генератор
    uniform_real_distribution<double> dist(0.0, 100.0); // распределение
    for(long long i=0;i<N;i++)              // заполняю массив
      data[i] = dist(gen);                  // случайное число
  }

  vector<int> counts(size, 0);              // сколько элементов каждому процессу
  vector<int> displs(size, 0);              // смещения для Scatterv

  long long base = N / size;                // базовый размер на процесс
  long long rem  = N % size;                // остаток
  for(int i=0;i<size;i++){                  // распределяю остаток по первым rem процессам
    long long cnt = base + (i < rem ? 1 : 0); // размер куска i-го процесса
    counts[i] = (int)cnt;                   // записываю count (int нужен MPI)
  }
  displs[0] = 0;                            // первое смещение 0
  for(int i=1;i<size;i++)                   // считаю displs
    displs[i] = displs[i-1] + counts[i-1];  // префикс-сумма counts

  int local_n = counts[rank];               // сколько элементов у этого процесса
  vector<double> local(local_n);            // локальный кусок

  MPI_Scatterv(                             
    rank==0 ? data.data() : nullptr,        // sendbuf только у rank 0
    counts.data(),                          // counts
    displs.data(),                          // displs
    MPI_DOUBLE,                             // тип данных
    local.data(),                           // recvbuf
    local_n,                                // сколько получить
    MPI_DOUBLE,                             // тип данных
    0,                                      // root
    MPI_COMM_WORLD                          // коммуникатор
  );                                        // распределяю массив с учетом остатка

  double local_sum = 0.0;                   // локальная сумма
  double local_sq  = 0.0;                   // локальная сумма квадратов
  for(int i=0;i<local_n;i++){               // проход по локальному массиву
    double x = local[i];                    // значение
    local_sum += x;                         // суммирую
    local_sq  += x * x;                     // суммирую квадраты
  }

  double global_sum = 0.0;                  // глобальная сумма
  double global_sq  = 0.0;                  // глобальная сумма квадратов

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // собираю суммы
  MPI_Reduce(&local_sq,  &global_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // собираю квадраты

  if(rank == 0){                            // считаю статистику на root
    double mean = global_sum / (double)N;   // среднее
    double var  = global_sq / (double)N - mean * mean; // дисперсия
    if(var < 0) var = 0;                    // защита от -0 из-за числ. ошибок
    double stdev = sqrt(var);               // стандартное отклонение

    cout.setf(std::ios::fixed);             // формат вывода
    cout << setprecision(6);                // точность
    cout << "N=" << N << "\n";              // печать N
    cout << "mean=" << mean << "\n";        // печать среднего
    cout << "stdev=" << stdev << "\n";      // печать std
  }

  MPI_Finalize();                           // завершаю MPI
  return 0;                                 // конец
}
