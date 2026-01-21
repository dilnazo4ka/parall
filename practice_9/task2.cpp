#include <bits/stdc++.h>          // стандартные C++ библиотеки
#include <mpi.h>                  // MPI
using namespace std;

// функция: определить, какой процесс владеет строкой r
int owner_of_row(int r, const vector<int>& start, const vector<int>& cnt){
  for(int p = 0; p < (int)start.size(); p++){
    if(r >= start[p] && r < start[p] + cnt[p]) return p;
  }
  return 0;
}

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);                         // инициализация MPI

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // номер процесса
  MPI_Comm_size(MPI_COMM_WORLD, &size);           // всего процессов

  int N = 128;                                    // размер матрицы
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

  vector<double> localA(local_rows * N);          // локальные строки A
  vector<double> localb(local_rows);              // локальный кусок b

  vector<double> A, b;                            // полные A и b (rank 0)

  if(rank == 0){
    A.resize(N * N);
    b.resize(N);
    mt19937 gen(42);
    uniform_real_distribution<double> dist(1.0, 10.0);

    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++)
        A[i*N + j] = dist(gen);
      A[i*N + i] += 100.0;                         // диагональное доминирование
      b[i] = dist(gen);
    }
  }

  // параметры Scatterv
  vector<int> scA(size), dspA(size), scb(size), dspb(size);
  for(int p = 0; p < size; p++){
    scA[p] = rows[p] * N;
    scb[p] = rows[p];
  }
  dspA[0] = dspb[0] = 0;
  for(int p = 1; p < size; p++){
    dspA[p] = dspA[p-1] + scA[p-1];
    dspb[p] = dspb[p-1] + scb[p-1];
  }

  MPI_Scatterv(rank==0?A.data():nullptr, scA.data(), dspA.data(), MPI_DOUBLE,
               localA.data(), local_rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank==0?b.data():nullptr, scb.data(), dspb.data(), MPI_DOUBLE,
               localb.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  vector<double> pivot_row(N);
  double pivot_b = 0.0;

  double t1 = MPI_Wtime();                         // старт таймера

  // ---------------- прямой ход Гаусса ----------------
  for(int k = 0; k < N; k++){
    int owner = owner_of_row(k, start, rows);

    if(rank == owner){
      int loc = k - start[rank];
      pivot_b = localb[loc];
      for(int j = 0; j < N; j++)
        pivot_row[j] = localA[loc*N + j];
    }

    MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for(int i = 0; i < local_rows; i++){
      int global_i = start[rank] + i;
      if(global_i > k){
        double factor = localA[i*N + k] / pivot_row[k];
        for(int j = k; j < N; j++)
          localA[i*N + j] -= factor * pivot_row[j];
        localb[i] -= factor * pivot_b;
      }
    }
  }

  double t2 = MPI_Wtime();                         // конец таймера

  if(rank == 0){
    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << "N=" << N << "\n";
    cout << "Gauss MPI time = " << (t2 - t1) << " s\n";
  }

  MPI_Finalize();                                  // завершение MPI
  return 0;
}
