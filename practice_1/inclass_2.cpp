// lab2_part2.cpp
// ========================================================================
// ЧАСТЬ 2: РАБОТА СО СТРУКТУРАМИ ДАННЫХ И ПАРАЛЛЕЛИЗАЦИЯ
// ========================================================================
// 
// Программа реализует три основные структуры данных:
// 1. ОДНОСВЯЗНЫЙ СПИСОК (Single Linked List)
// 2. СТЕК (Stack) - LIFO
// 3. ОЧЕРЕДЬ (Queue) - FIFO
//
// Для каждой структуры сравнивается:
// - Последовательное добавление элементов
// - Параллельное добавление элементов с OpenMP
//
// ========================================================================
// СБОРКА:
// ========================================================================
// g++ -fopenmp -O2 -std=c++17 lab2_part2.cpp -o lab2_part2.exe
//
// ========================================================================
// ЗАПУСК:
// ========================================================================
// $env:OMP_NUM_THREADS = '4'
// .\lab2_part2.exe
// ========================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <random>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ========================================================================
// 1. ОДНОСВЯЗНЫЙ СПИСОК (Single Linked List)
// ========================================================================
template <typename T>
class LinkedListNode {
public:
    T data;
    LinkedListNode* next;
    
    LinkedListNode(T val) : data(val), next(nullptr) {}
};

template <typename T>
class LinkedList {
private:
    LinkedListNode<T>* head;
    mutex mtx;
    
public:
    LinkedList() : head(nullptr) {}
    
    // Деструктор: освобождает память
    ~LinkedList() {
        while (head) {
            LinkedListNode<T>* temp = head;
            head = head->next;
            delete temp;
        }
    }
    
    // Добавление элемента в начало (O(1))
    void addFront(T val) {
        lock_guard<mutex> lock(mtx);
        LinkedListNode<T>* newNode = new LinkedListNode<T>(val);
        newNode->next = head;
        head = newNode;
    }
    
    // Удаление элемента с начала (O(1))
    bool removeFront() {
        lock_guard<mutex> lock(mtx);
        if (!head) return false;
        LinkedListNode<T>* temp = head;
        head = head->next;
        delete temp;
        return true;
    }
    
    // Поиск элемента (O(n))
    bool search(T val) {
        lock_guard<mutex> lock(mtx);
        LinkedListNode<T>* current = head;
        while (current) {
            if (current->data == val) return true;
            current = current->next;
        }
        return false;
    }
    
    // Получение размера списка (O(n))
    int size() {
        lock_guard<mutex> lock(mtx);
        int count = 0;
        LinkedListNode<T>* current = head;
        while (current) {
            count++;
            current = current->next;
        }
        return count;
    }
    
    // Печать всех элементов
    void print() {
        lock_guard<mutex> lock(mtx);
        LinkedListNode<T>* current = head;
        cout << "List: ";
        int count = 0;
        while (current && count < 20) {
            cout << current->data << " -> ";
            current = current->next;
            count++;
        }
        if (current) cout << "...";
        cout << "nullptr\n";
    }
};

// ========================================================================
// 2. СТЕК (Stack) - LIFO
// ========================================================================
template <typename T>
class Stack {
private:
    vector<T> data;
    mutex mtx;
    
public:
    // Добавление элемента (O(1))
    void push(T val) {
        lock_guard<mutex> lock(mtx);
        data.push_back(val);
    }
    
    // Удаление элемента (O(1))
    bool pop() {
        lock_guard<mutex> lock(mtx);
        if (data.empty()) return false;
        data.pop_back();
        return true;
    }
    
    // Получение верхнего элемента
    bool top(T& val) {
        lock_guard<mutex> lock(mtx);
        if (data.empty()) return false;
        val = data.back();
        return true;
    }
    
    // Проверка на пустоту
    bool isEmpty() {
        lock_guard<mutex> lock(mtx);
        return data.empty();
    }
    
    // Размер стека
    int size() {
        lock_guard<mutex> lock(mtx);
        return data.size();
    }
    
    // Печать стека (от вершины к основанию)
    void print() {
        lock_guard<mutex> lock(mtx);
        cout << "Stack (top to bottom): ";
        int count = 0;
        for (int i = data.size() - 1; i >= 0 && count < 20; --i, ++count) {
            cout << data[i] << " ";
        }
        if (data.size() > 20) cout << "...";
        cout << "\n";
    }
};

// ========================================================================
// 3. ОЧЕРЕДЬ (Queue) - FIFO
// ========================================================================
template <typename T>
class Queue {
private:
    vector<T> data;
    size_t front_idx;
    mutex mtx;
    
public:
    Queue() : front_idx(0) {}
    
    // Добавление в конец (O(1))
    void enqueue(T val) {
        lock_guard<mutex> lock(mtx);
        data.push_back(val);
    }
    
    // Удаление из начала (O(1) амортизированная)
    bool dequeue() {
        lock_guard<mutex> lock(mtx);
        if (front_idx >= data.size()) return false;
        front_idx++;
        
        // Оптимизация памяти
        if (front_idx > data.size() / 2 && front_idx > 100) {
            data.erase(data.begin(), data.begin() + front_idx);
            front_idx = 0;
        }
        return true;
    }
    
    // Получение первого элемента
    bool front(T& val) {
        lock_guard<mutex> lock(mtx);
        if (front_idx >= data.size()) return false;
        val = data[front_idx];
        return true;
    }
    
    // Проверка на пустоту
    bool isEmpty() {
        lock_guard<mutex> lock(mtx);
        return front_idx >= data.size();
    }
    
    // Размер очереди
    int size() {
        lock_guard<mutex> lock(mtx);
        return max(0, (int)data.size() - (int)front_idx);
    }
    
    // Печать очереди (от начала к концу)
    void print() {
        lock_guard<mutex> lock(mtx);
        cout << "Queue (front to back): ";
        int count = 0;
        for (size_t i = front_idx; i < data.size() && count < 20; ++i, ++count) {
            cout << data[i] << " ";
        }
        if (data.size() - front_idx > 20) cout << "...";
        cout << "\n";
    }
};

// ========================================================================
// ТЕСТИРОВАНИЕ: ОДНОСВЯЗНЫЙ СПИСОК
// ========================================================================
void testLinkedList(int N, int threads) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  TEST: Linked List (Single Linked List)\n";
    cout << "=================================================================\n";
    cout << "Task: Add " << N << " elements\n";
    cout << "Threads: " << threads << "\n";
    cout << "-----------------------------------------------------------------\n";
    
    // ПОСЛЕДОВАТЕЛЬНОЕ добавление
    LinkedList<int> listSeq;
    cout << "\n[1] SEQUENTIAL ADD\n";
    
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        listSeq.addFront(i);
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_time = t2 - t1;
    
    cout << "    Elements added: " << listSeq.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << seq_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (listSeq.size() / seq_time.count()) << " elem/ms\n";
    
    // ПАРАЛЛЕЛЬНОЕ добавление
    LinkedList<int> listPar;
    cout << "\n[2] PARALLEL ADD\n";
    
    auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
    omp_set_num_threads(threads);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        listPar.addFront(i);
    }
#else
    for (int i = 0; i < N; ++i) {
        listPar.addFront(i);
    }
#endif
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> par_time = t4 - t3;
    
    cout << "    Elements added: " << listPar.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << par_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (listPar.size() / par_time.count()) << " elem/ms\n";
    
    // АНАЛИЗ
    cout << "\n[3] ANALYSIS\n";
    double speedup = seq_time.count() / par_time.count();
    cout << "    Speedup: " << fixed << setprecision(3) << speedup << "x\n";
    cout << "    Efficiency: " << fixed << setprecision(2) 
         << (speedup / threads * 100) << "%\n";
    
    if (speedup > 1.0) {
        cout << "    → Parallel is FASTER\n";
    } else {
        cout << "    → Sequential is FASTER (overhead dominates)\n";
    }
    
    // ДЕМОНСТРАЦИЯ операций
    if (N <= 10) {
        cout << "\n[4] DEMONSTRATION\n";
        cout << "    Sequential list:\n    ";
        listSeq.print();
        cout << "    Parallel list:\n    ";
        listPar.print();
        
        // Поиск элемента
        int search_val = N / 2;
        cout << "\n    Search for " << search_val << ":\n";
        cout << "      Sequential: " << (listSeq.search(search_val) ? "Found" : "Not found") << "\n";
        cout << "      Parallel: " << (listPar.search(search_val) ? "Found" : "Not found") << "\n";
        
        // Удаление элемента
        cout << "\n    Removing front element:\n";
        listSeq.removeFront();
        listPar.removeFront();
        cout << "      Sequential size after removal: " << listSeq.size() << "\n";
        cout << "      Parallel size after removal: " << listPar.size() << "\n";
    }
    
    cout << "=================================================================\n";
}

// ========================================================================
// ТЕСТИРОВАНИЕ: СТЕК
// ========================================================================
void testStack(int N, int threads) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  TEST: Stack (LIFO - Last In First Out)\n";
    cout << "=================================================================\n";
    cout << "Task: Push " << N << " elements\n";
    cout << "Threads: " << threads << "\n";
    cout << "-----------------------------------------------------------------\n";
    
    // ПОСЛЕДОВАТЕЛЬНОЕ добавление
    Stack<int> stackSeq;
    cout << "\n[1] SEQUENTIAL PUSH\n";
    
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        stackSeq.push(i);
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_time = t2 - t1;
    
    cout << "    Elements pushed: " << stackSeq.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << seq_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (stackSeq.size() / seq_time.count()) << " elem/ms\n";
    
    // ПАРАЛЛЕЛЬНОЕ добавление
    Stack<int> stackPar;
    cout << "\n[2] PARALLEL PUSH\n";
    
    auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
    omp_set_num_threads(threads);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        stackPar.push(i);
    }
#else
    for (int i = 0; i < N; ++i) {
        stackPar.push(i);
    }
#endif
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> par_time = t4 - t3;
    
    cout << "    Elements pushed: " << stackPar.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << par_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (stackPar.size() / par_time.count()) << " elem/ms\n";
    
    // АНАЛИЗ
    cout << "\n[3] ANALYSIS\n";
    double speedup = seq_time.count() / par_time.count();
    cout << "    Speedup: " << fixed << setprecision(3) << speedup << "x\n";
    cout << "    Efficiency: " << fixed << setprecision(2) 
         << (speedup / threads * 100) << "%\n";
    
    if (speedup > 1.0) {
        cout << "    → Parallel is FASTER\n";
    } else {
        cout << "    → Sequential is FASTER (overhead dominates)\n";
    }
    
    // ДЕМОНСТРАЦИЯ операций
    if (N <= 10) {
        cout << "\n[4] DEMONSTRATION\n";
        cout << "    Sequential stack:\n    ";
        stackSeq.print();
        cout << "    Parallel stack:\n    ";
        stackPar.print();
        
        // Pop операции
        cout << "\n    Popping top element:\n";
        int val;
        if (stackSeq.top(val)) {
            cout << "      Sequential top value: " << val << "\n";
        }
        stackSeq.pop();
        
        if (stackPar.top(val)) {
            cout << "      Parallel top value: " << val << "\n";
        }
        stackPar.pop();
        
        cout << "      Sequential size after pop: " << stackSeq.size() << "\n";
        cout << "      Parallel size after pop: " << stackPar.size() << "\n";
    }
    
    cout << "=================================================================\n";
}

// ========================================================================
// ТЕСТИРОВАНИЕ: ОЧЕРЕДЬ
// ========================================================================
void testQueue(int N, int threads) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  TEST: Queue (FIFO - First In First Out)\n";
    cout << "=================================================================\n";
    cout << "Task: Enqueue " << N << " elements\n";
    cout << "Threads: " << threads << "\n";
    cout << "-----------------------------------------------------------------\n";
    
    // ПОСЛЕДОВАТЕЛЬНОЕ добавление
    Queue<int> queueSeq;
    cout << "\n[1] SEQUENTIAL ENQUEUE\n";
    
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        queueSeq.enqueue(i);
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seq_time = t2 - t1;
    
    cout << "    Elements enqueued: " << queueSeq.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << seq_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (queueSeq.size() / seq_time.count()) << " elem/ms\n";
    
    // ПАРАЛЛЕЛЬНОЕ добавление
    Queue<int> queuePar;
    cout << "\n[2] PARALLEL ENQUEUE\n";
    
    auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
    omp_set_num_threads(threads);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        queuePar.enqueue(i);
    }
#else
    for (int i = 0; i < N; ++i) {
        queuePar.enqueue(i);
    }
#endif
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> par_time = t4 - t3;
    
    cout << "    Elements enqueued: " << queuePar.size() << "\n";
    cout << "    Time: " << fixed << setprecision(6) << par_time.count() << " ms\n";
    cout << "    Rate: " << fixed << setprecision(2) 
         << (queuePar.size() / par_time.count()) << " elem/ms\n";
    
    // АНАЛИЗ
    cout << "\n[3] ANALYSIS\n";
    double speedup = seq_time.count() / par_time.count();
    cout << "    Speedup: " << fixed << setprecision(3) << speedup << "x\n";
    cout << "    Efficiency: " << fixed << setprecision(2) 
         << (speedup / threads * 100) << "%\n";
    
    if (speedup > 1.0) {
        cout << "    → Parallel is FASTER\n";
    } else {
        cout << "    → Sequential is FASTER (overhead dominates)\n";
    }
    
    // ДЕМОНСТРАЦИЯ операций
    if (N <= 10) {
        cout << "\n[4] DEMONSTRATION\n";
        cout << "    Sequential queue:\n    ";
        queueSeq.print();
        cout << "    Parallel queue:\n    ";
        queuePar.print();
        
        // Dequeue операции
        cout << "\n    Dequeuing front element:\n";
        int val;
        if (queueSeq.front(val)) {
            cout << "      Sequential front value: " << val << "\n";
        }
        queueSeq.dequeue();
        
        if (queuePar.front(val)) {
            cout << "      Parallel front value: " << val << "\n";
        }
        queuePar.dequeue();
        
        cout << "      Sequential size after dequeue: " << queueSeq.size() << "\n";
        cout << "      Parallel size after dequeue: " << queuePar.size() << "\n";
    }
    
    cout << "=================================================================\n";
}

// ========================================================================
// ГЛАВНАЯ ФУНКЦИЯ
// ========================================================================
int main() {
    cout << "\n";
    cout << "=================================================================\n";
    cout << "    LAB 2 - PART 2: Data Structures with OpenMP\n";
    cout << "=================================================================\n";
    
    // Проверка OpenMP
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
    cout << "[OpenMP] Available\n";
    cout << "[OpenMP] Max threads: " << num_threads << "\n";
#else
    cout << "[OpenMP] Not available (compile with -fopenmp)\n";
#endif
    
    cout << "\nProgram will test:\n";
    cout << "  1. Linked List (dynamic pointer-based structure)\n";
    cout << "  2. Stack (LIFO structure)\n";
    cout << "  3. Queue (FIFO structure)\n";
    
    // Размеры для тестирования
    vector<int> test_sizes = {10, 10000, 100000, 1000000};
    
    cout << "\nTest sizes: ";
    for (size_t i = 0; i < test_sizes.size(); ++i) {
        cout << test_sizes[i];
        if (i < test_sizes.size() - 1) cout << ", ";
    }
    cout << "\n";
    
    // Запускаем тесты
    for (int N : test_sizes) {
        cout << "\n-----------------------------------------------------------------\n";
        cout << "  ROUND: N = " << N << " elements\n";
        cout << "-----------------------------------------------------------------\n";
        
        testLinkedList(N, num_threads);
        testStack(N, num_threads);
        testQueue(N, num_threads);
    }
    
    // Итоговый анализ
    cout << "\n";
    cout << "=================================================================\n";
    cout << "  SUMMARY & CONCLUSIONS\n";
    cout << "=================================================================\n";
    cout << "\n1. LINKED LIST:\n";
    cout << "   - Dynamic memory allocation for each node\n";
    cout << "   - Mutex protects shared structure\n";
    cout << "   - Parallel speedup depends on N and thread overhead\n";
    cout << "\n2. STACK (LIFO):\n";
    cout << "   - Vector-based implementation\n";
    cout << "   - Fast push/pop operations O(1)\n";
    cout << "   - Parallel push benefits from multiple threads\n";
    cout << "\n3. QUEUE (FIFO):\n";
    cout << "   - Vector-based with front index optimization\n";
    cout << "   - Enqueue is easily parallelizable\n";
    cout << "   - Dequeue typically sequential (FIFO order)\n";
    cout << "\n4. PERFORMANCE INSIGHTS:\n";
    cout << "   - Small N (< 10,000): sequential often faster\n";
    cout << "   - Large N (> 100,000): parallel shows benefits\n";
    cout << "   - Mutex contention can limit scalability\n";
    cout << "   - Thread overhead vs. actual work is crucial\n";
    cout << "=================================================================\n";
    
    cout << "\n[Program] All tests completed successfully!\n\n";
    
    return 0;
}