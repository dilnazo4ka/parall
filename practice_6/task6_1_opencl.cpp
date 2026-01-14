#include <CL/cl.h>                         // подключаю OpenCL API
#include <iostream>                        // для cout/cerr
#include <vector>                          // для std::vector
#include <fstream>                         // чтобы читать kernel.cl
#include <sstream>                         // чтобы собрать текст ядра
#include <chrono>                          // для измерения времени
#include <cmath>                           // для fabs

using namespace std;                       // чтобы не писать std::

#define CHECK_CL(err, msg) do {            \
  if ((err) != CL_SUCCESS) {               \
    cerr << "OpenCL error (" << (err)      \
         << ") at: " << (msg) << "\n";     \
    exit(1);                               \
  }                                        \
} while(0)

// ------------------------------
// функция: читаю kernel.cl в строку
// ------------------------------
static string load_text_file(const string& path) {          // объявляю функцию чтения файла
  ifstream f(path);                                         // открываю файл
  if (!f.is_open()) {                                       // проверяю открылся ли
    cerr << "Cannot open file: " << path << "\n";            // печатаю ошибку
    exit(1);                                                // выхожу
  }
  stringstream ss;                                          // создаю поток-буфер
  ss << f.rdbuf();                                          // читаю весь файл
  return ss.str();                                          // возвращаю содержимое
}

// ------------------------------
// функция: выбрать устройство (CPU или GPU)
// ------------------------------
static cl_device_id pick_device(cl_device_type dtype) {     // dtype = CL_DEVICE_TYPE_CPU или GPU
  cl_int err;                                               // переменная для ошибок

  cl_uint num_platforms = 0;                                // количество платформ
  CHECK_CL(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)"); // узнаю число платформ
  if (num_platforms == 0) {                                 // если платформ нет
    cerr << "No OpenCL platforms found.\n";                  // печатаю
    exit(1);                                                // выхожу
  }

  vector<cl_platform_id> plats(num_platforms);              // массив платформ
  CHECK_CL(clGetPlatformIDs(num_platforms, plats.data(), nullptr), "clGetPlatformIDs(list)"); // получаю платформы

  // пробую найти устройство нужного типа на любой платформе
  for (auto p : plats) {                                    // иду по платформам
    cl_uint num_devs = 0;                                   // число устройств
    err = clGetDeviceIDs(p, dtype, 0, nullptr, &num_devs);   // пытаюсь узнать количество устройств нужного типа
    if (err != CL_SUCCESS || num_devs == 0) continue;        // если нет — пропускаю платформу

    vector<cl_device_id> devs(num_devs);                    // массив устройств
    CHECK_CL(clGetDeviceIDs(p, dtype, num_devs, devs.data(), nullptr), "clGetDeviceIDs(list)"); // получаю устройства
    return devs[0];                                         // беру первое устройство
  }

  return nullptr;                                           // если не нашли — вернём nullptr
}

// ------------------------------
// функция: выполнить vector_add на конкретном устройстве и вернуть время (мс)
// ------------------------------
static double run_vector_add(cl_device_type dtype, const vector<float>& A, const vector<float>& B, vector<float>& C) {
  cl_int err;                                               // для ошибок
  size_t N = A.size();                                      // размер массива
  size_t bytes = N * sizeof(float);                         // сколько байт

  cl_device_id dev = pick_device(dtype);                    // выбираю устройство
  if (!dev) {                                               // если не нашли
    cerr << "Requested device type not found.\n";            // печатаю
    return -1.0;                                            // возвращаю -1
  }

  // (не обязательно) выведу имя устройства
  char name[256];                                           // буфер для имени
  CHECK_CL(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo(NAME)");
  cout << "Device: " << name << "\n";                        // печатаю имя

  // создаю контекст (связка устройства и ресурсов)
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  CHECK_CL(err, "clCreateContext");

  // создаю командную очередь (куда отправляются команды)
  // для профилирования включаю CL_QUEUE_PROFILING_ENABLE
  cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_CL(err, "clCreateCommandQueue");

  // читаю текст ядра из kernel.cl
  string src = load_text_file("kernel.cl");                 // читаю файл ядра
  const char* csrc = src.c_str();                            // c-string
  size_t srclen = src.size();                                // длина

  // создаю и собираю программу
  cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
  CHECK_CL(err, "clCreateProgramWithSource");

  err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr); // компиляция программы
  if (err != CL_SUCCESS) {                                   // если компиляция не удалась
    // читаю лог компиляции
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    string log(log_size, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    cerr << "Build log:\n" << log << "\n";
    CHECK_CL(err, "clBuildProgram");
  }

  // создаю kernel
  cl_kernel kernel = clCreateKernel(prog, "vector_add", &err);
  CHECK_CL(err, "clCreateKernel");

  // создаю буферы в глобальной памяти устройства
  cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  CHECK_CL(err, "clCreateBuffer(A)");
  cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
  CHECK_CL(err, "clCreateBuffer(B)");
  cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
  CHECK_CL(err, "clCreateBuffer(C)");

  // копирую данные на устройство
  CHECK_CL(clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr), "WriteBuffer(A)");
  CHECK_CL(clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr), "WriteBuffer(B)");

  // ставлю аргументы ядра
  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA), "clSetKernelArg(0)");
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB), "clSetKernelArg(1)");
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC), "clSetKernelArg(2)");

  // задаю размеры выполнения
  size_t global = N;                                         // глобальный размер = N потоков
  size_t local  = 256;                                       // локальный размер (work-group)
  if (global % local != 0) {                                 // если не делится
    global = ((global + local - 1) / local) * local;          // округляю вверх до кратного local
  }

  // запускаю kernel + меряю через profiling event
  cl_event evt;                                              // event для профилирования
  CHECK_CL(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, &evt), "clEnqueueNDRangeKernel");

  CHECK_CL(clFinish(q), "clFinish");                          // жду завершения всех команд

  // беру время исполнения kernel из профилирования
  cl_ulong t0 = 0, t1 = 0;                                    // timestamps
  CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "Profiling START");
  CHECK_CL(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "Profiling END");

  double kernel_ms = (t1 - t0) * 1e-6;                        // наносекунды -> миллисекунды

  // читаю результат обратно
  CHECK_CL(clEnqueueReadBuffer(q, dC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr), "ReadBuffer(C)");

  // освобождаю ресурсы
  clReleaseEvent(evt);                                        // освобождаю event
  clReleaseMemObject(dA);                                     // освобождаю буфер A
  clReleaseMemObject(dB);                                     // освобождаю буфер B
  clReleaseMemObject(dC);                                     // освобождаю буфер C
  clReleaseKernel(kernel);                                    // освобождаю kernel
  clReleaseProgram(prog);                                     // освобождаю программу
  clReleaseCommandQueue(q);                                   // освобождаю очередь
  clReleaseContext(ctx);                                      // освобождаю контекст

  return kernel_ms;                                           // возвращаю время kernel (мс)
}

int main() {
  const size_t N = 1 << 20;                                  // беру N = 1,048,576 (удобно для GPU)
  vector<float> A(N), B(N), C(N);                            // создаю массивы

  // заполняю данными
  for (size_t i = 0; i < N; i++) {                           // иду по массивам
    A[i] = float(i) * 0.001f;                                // A[i]
    B[i] = float(i) * 0.002f;                                // B[i]
  }

  cout << "N=" << N << "\n";                                 // печатаю размер

  // запуск на CPU
  double cpu_ms = run_vector_add(CL_DEVICE_TYPE_CPU, A, B, C); // запускаю на CPU-устройстве OpenCL
  if (cpu_ms >= 0) cout << "CPU kernel time (ms): " << cpu_ms << "\n"; // печатаю время

  // запуск на GPU
  double gpu_ms = run_vector_add(CL_DEVICE_TYPE_GPU, A, B, C); // запускаю на GPU-устройстве OpenCL
  if (gpu_ms >= 0) cout << "GPU kernel time (ms): " << gpu_ms << "\n"; // печатаю время

  // проверка корректности (пару элементов)
  bool ok = true;                                             // флаг корректности
  for (int i = 0; i < 10; i++) {                               // проверю первые 10
    float ref = A[i] + B[i];                                   // эталон
    if (fabs(C[i] - ref) > 1e-5f) ok = false;                  // сравниваю
  }
  cout << "Check: " << (ok ? "OK" : "FAIL") << "\n";           // печатаю результат

  return 0;                                                    // конец
}
