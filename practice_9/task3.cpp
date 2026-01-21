#include <bits/stdc++.h>          // стандартные C++ библиотеки
#include <mpi.h>                  // MPI
using namespace std;

const double INF = 1e12;          // большое число (бесконечность)

// определяю, какой процесс владеет строкой k
int owner_of_row(int k, const vector<int>& start, const vector<int>& cnt){
  for(int p = 0; p < (int)start.size(); p++){
    if(k >= start[p] && k < start[p] + cnt[p]) return p;
  }
  return 0;
}

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);                         // инициализация MPI

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // номер процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size);           // всего процессов

  int N = 128;                                    // размер графа
  if(argc > 1) N = atoi(argv[1]);                 // можно передать N аргументом

  // ---------------- распределение строк ----------------
  vector<int> rows(size), start(size);
  int base = N / size, rem = N % size;
  for(int p = 0; p < size; p++)
    rows[p] = base + (p < rem ? 1 : 0);

  start[0] = 0;
  for(int p = 1; p < size; p++)
    start[p] = start[p-1] + rows[p-1];

  int local_rows = rows[rank];                    // строки текущего процесса

  vector<double> localD(local_rows * N);          // локальный блок матрицы
  vector<double> D;                               // полная матрица (rank 0)

  if(rank == 0){
    D.resize(N * N);
    mt19937 gen(42);
    uniform_real_distribution<double> dist(1.0, 20.0);

    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++){
        if(i == j) D[i*N + j] = 0.0;
        else if(gen() % 4 == 0) D[i*N + j] = INF; // нет ребра
        else D[i*N + j] = dist(gen);
      }
    }
  }

  // параметры Scatterv / Gatherv
  vector<int> sc(size), dsp(size);
  for(int p = 0; p < size; p++)
    sc[p] = rows[p] * N;

  dsp[0] = 0;
  for(int p = 1; p < size; p++)
    dsp[p] = dsp[p-1] + sc[p-1];

  MPI_Scatterv(rank==0?D.data():nullptr, sc.data(), dsp.data(), MPI_DOUBLE,
               localD.data(), local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  vector<double> kth_row(N);                      // буфер строки k

  double t1 = MPI_Wtime();                        // старт времени

  // ---------------- алгоритм Флойда–Уоршелла ----------------
  for(int k = 0; k < N; k++){
    int owner = owner_of_row(k, start, rows);

    if(rank == owner){
      int loc = k - start[rank];
      for(int j = 0; j < N; j++)
        kth_row[j] = localD[loc*N + j];
    }

    MPI_Bcast(kth_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for(int i = 0; i < local_rows; i++){
      for(int j = 0; j < N; j++){
        double via_k = localD[i*N + k] + kth_row[j];
        if(via_k < localD[i*N + j])
          localD[i*N + j] = via_k;
      }
    }
  }

  double t2 = MPI_Wtime();                        // конец времени

  MPI_Gatherv(localD.data(), local_rows*N, MPI_DOUBLE,
              rank==0?D.data():nullptr, sc.data(), dsp.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(rank == 0){
    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << "N=" << N << "\n";
    cout << "Floyd-Warshall MPI time = " << (t2 - t1) << " s\n";
    cout << "check D[0][1] = " << D[1] << "\n";  // быстрая проверка
  }

  MPI_Finalize();                                 // завершение MPI
  return 0;
}
