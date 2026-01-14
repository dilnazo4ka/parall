#define CL_TARGET_OPENCL_VERSION 120                       // фиксирую версию OpenCL 1.2, чтобы не было info-спама
#include <CL/cl.h>                                         // подключаю OpenCL API

#include <iostream>                                        // для cout/cerr
#include <vector>                                          // для vector
#include <fstream>                                         // для чтения .cl файла
#include <sstream>                                         // для stringstream
#include <chrono>                                          // для CPU тайминга
#include <cmath>                                           // для fabs

using namespace std;                                       // чтобы не писать std::

#define CHECK_CL(err, msg) do {                            \
  if ((err) != CL_SUCCESS) {                               \
    cerr << "OpenCL error (" << (err)                      \
         << ") at: " << (msg) << "\n";                     \
    exit(1);                                               \
  }                                                        \
} while(0)

// читаю текст kernel-файла в строку
static string load_text_file(const string& path) {         // функция чтения файла
  ifstream f(path);                                        // открываю файл
  if (!f.is_open()) {                                      // если не открылся
    cerr << "Cannot open file: " << path << "\n";           // печатаю ошибку
    exit(1);                                               // завершаю
  }
  stringstream ss;                                         // буфер
  ss << f.rdbuf();                                         // читаю всё содержимое
  return ss.str();                                         // возвращаю строку
}

// выбираю устройство нужного типа (CPU или GPU)
static cl_device_id pick_device(cl_device_type dtype) {    // dtype = CL_DEVICE_TYPE_CPU / GPU
  cl_uint num_platforms = 0;                               // число платформ
  CHECK_CL(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)"); // получаю число платформ
  if (num_platforms == 0) {                                // если платформ нет
    return nullptr;                                        // возвращаю nullptr
  }

  vector<cl_platform_id> plats(num_platforms);             // массив платформ
  CHECK_CL(clGetPlatformIDs(num_platforms, plats.data(), nullptr), "clGetPlatformIDs(list)"); // список платформ

  for (auto p : plats) {                                   // прохожу по платформам
    cl_uint num_devs = 0;                                  // число устройств
    cl_int err = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs); // пытаюсь найти устройства нужного типа
    if (err != CL_SUCCESS || num_devs == 0) continue;       // если нет — следующая платформа

    vector<cl_device_id> devs(num_devs);                   // массив устройств
    CHECK_CL(clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr), "clGetDeviceIDs(list)"); // получаю девайсы
    return devs[0];                                        // беру первое устройство
  }

  return nullptr;                                          // если не нашли
}

// последовательное матричное умножение на CPU для проверки
static void cpu_matmul(const vector<float>& A, const vector<float>& B, vector<float>& C,
                       int N, int M, int K) {              // N×M * M×K = N×K
  for (int i = 0; i < N; i++) {                            // строки
    for (int j = 0; j < K; j++) {                          // столбцы
      double sum = 0.0;                                    // double для точности эталона
      for (int t = 0; t < M; t++) {                        // общая размерность
        sum += (double)A[i*M + t] * (double)B[t*K + j];    // суммирую произведения
      }
      C[i*K + j] = (float)sum;                             // сохраняю
    }
  }
}

// запуск OpenCL matmul на CPU или GPU, возвращает время ядра (мс)
static double run_opencl_matmul(cl_device_type dtype,
                                const vector<float>& A,
                                const vector<float>& B,
                                vector<float>& C,
                                int N, int M, int K) {
  cl_device_id dev = pick_device(dtype);                   // выбираю девайс
  if (!dev) {                                              // если не найден
    cerr << "Requested device type not found.\n";           // сообщаю
    return -1.0;                                           // возвращаю -1
  }

  char name[256];                                          // буфер для имени
  CHECK_CL(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo(NAME)"); // имя
  cout << "Device: " << name << "\n";                      // печатаю имя

  cl_int err;                                              // переменная ошибок

  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); // создаю контекст
  CHECK_CL(err, "clCreateContext");

  // очередь с profiling (как в task1)
  cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); // очередь
  CHECK_CL(err, "clCreateCommandQueue");

  // читаю ядро
  string src = load_text_file("matmul.cl");                // текст ядра
  const char* csrc = src.c_str();                          // char*
  size_t srclen = src.size();                              // длина

  cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err); // программа из исходника
  CHECK_CL(err, "clCreateProgramWithSource");

  err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);            // сборка
  if (err != CL_SUCCESS) {                                                    // если ошибка сборки
    size_t log_size = 0;                                                      // размер лога
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size); // узнаю размер
    string log(log_size, '\0');                                               // строка под лог
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr); // читаю лог
    cerr << "Build log:\n" << log << "\n";                                     // печатаю лог
    CHECK_CL(err, "clBuildProgram");                                           // падаю
  }

  cl_kernel kernel = clCreateKernel(prog, "matmul", &err);   // создаю kernel
  CHECK_CL(err, "clCreateKernel");

  size_t bytesA = (size_t)N * M * sizeof(float);             // байты для A
  size_t bytesB = (size_t)M * K * sizeof(float);             // байты для B
  size_t bytesC = (size_t)N * K * sizeof(float);             // байты для C

  cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesA, nullptr, &err);  // буфер A
  CHECK_CL(err, "clCreateBuffer(A)");
  cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesB, nullptr, &err);  // буфер B
  CHECK_CL(err, "clCreateBuffer(B)");
  cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesC, nullptr, &err); // буфер C
  CHECK_CL(err, "clCreateBuffer(C)");

  CHECK_CL(clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, bytesA, A.data(), 0, nullptr, nullptr), "WriteBuffer(A)"); // копирую A
  CHECK_CL(clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, bytesB, B.data(), 0, nullptr, nullptr), "WriteBuffer(B)"); // копирую B

  // ставлю аргументы ядра (A,B,C,N,M,K)
  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA), "SetArg(0)");      // A
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB), "SetArg(1)");      // B
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC), "SetArg(2)");      // C
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(int), &N), "SetArg(3)");          // N
  CHECK_CL(clSetKernelArg(kernel, 4, sizeof(int), &M), "SetArg(4)");          // M
  CHECK_CL(clSetKernelArg(kernel, 5, sizeof(int), &K), "SetArg(5)");          // K

  // 2D NDRange: (row, col) для матрицы C размера N×K
  size_t global[2] = { (size_t)N, (size_t)K };            // глобальный размер по двум измерениям
  size_t local[2]  = { 16, 16 };                          // локальная группа 16×16 (часто норм для матриц)

  // округляю global вверх до кратности local, чтобы было безопасно
  global[0] = ((global[0] + local[0] - 1) / local[0]) * local[0]; // округление N
  global[1] = ((global[1] + local[1] - 1) / local[1]) * local[1]; // округление K

  cl_event evt;                                           // event для профилирования
  CHECK_CL(clEnqueueNDRangeKernel(q, kernel, 2, nullptr, global, local, 0, nullptr, &evt), "EnqueueKernel"); // запускаю
  CHECK_CL(clFinish(q), "clFinish");                       // жду завершения

  // беру профилировочные timestamps
  cl_ulong t0 = 0, t1 = 0;                                 // start/end
  CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "Profiling START"); // старт
  CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "Profiling END");     // конец

  double kernel_ms = (t1 - t0) * 1e-6;                     // наносекунды -> миллисекунды

  CHECK_CL(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, bytesC, C.data(), 0, nullptr, nullptr), "ReadBuffer(C)"); // читаю C

  // освобождаю ресурсы
  clReleaseEvent(evt);                                     // event
  clReleaseMemObject(dA);                                  // буферы
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseKernel(kernel);                                 // kernel
  clReleaseProgram(prog);                                  // program
  clReleaseCommandQueue(q);                                // queue
  clReleaseContext(ctx);                                   // context

  return kernel_ms;                                        // возвращаю время kernel
}

int main() {
  // размеры матриц (можно менять)
  // A: N×M, B: M×K, C: N×K
  int N = 256;                                             // строки A
  int M = 256;                                             // общая размерность
  int K = 256;                                             // столбцы B

  cout << "Matrix sizes: A(" << N << "x" << M << "), "
       << "B(" << M << "x" << K << "), "
       << "C(" << N << "x" << K << ")\n";

  vector<float> A((size_t)N * M);                          // матрица A
  vector<float> B((size_t)M * K);                          // матрица B
  vector<float> C((size_t)N * K);                          // результат OpenCL
  vector<float> Cref((size_t)N * K);                       // эталон CPU

  // заполняю матрицы
  for (int i = 0; i < N*M; i++) A[i] = (float)((i % 100) * 0.01f); // A
  for (int i = 0; i < M*K; i++) B[i] = (float)((i % 100) * 0.02f); // B

  // CPU reference + time
  auto t1 = chrono::high_resolution_clock::now();          // старт CPU
  cpu_matmul(A, B, Cref, N, M, K);                         // CPU умножение
  auto t2 = chrono::high_resolution_clock::now();          // конец CPU
  double cpu_ms = chrono::duration<double, milli>(t2 - t1).count(); // CPU время

  cout << "CPU reference time (ms): " << cpu_ms << "\n";   // печатаю CPU время

  // OpenCL на CPU
  double ocl_cpu_ms = run_opencl_matmul(CL_DEVICE_TYPE_CPU, A, B, C, N, M, K); // OpenCL CPU
  if (ocl_cpu_ms >= 0) cout << "OpenCL CPU kernel time (ms): " << ocl_cpu_ms << "\n"; // печатаю

  // OpenCL на GPU
  double ocl_gpu_ms = run_opencl_matmul(CL_DEVICE_TYPE_GPU, A, B, C, N, M, K); // OpenCL GPU
  if (ocl_gpu_ms >= 0) cout << "OpenCL GPU kernel time (ms): " << ocl_gpu_ms << "\n"; // печатаю

  // проверка корректности (сравнение с CPU эталоном)
  double max_err = 0.0;                                    // максимальная ошибка
  for (int i = 0; i < N*K; i++) {                          // прохожу по C
    max_err = max(max_err, (double)fabs(C[i] - Cref[i]));  // ищу max |C - Cref|
  }
  cout << "Max error: " << max_err << "\n";                // печатаю ошибку
  cout << "Check: " << ((max_err < 1e-3) ? "OK" : "FAIL") << "\n"; // итог

  return 0;                                                // конец
}
