#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

vector<int> generate_array(int n, uint64_t seed) {
    mt19937_64 gen(seed);
    uniform_int_distribution<int> dist(0, 100000);
    vector<int> a(n);
    for (int &x : a) x = dist(gen);
    return a;
}

bool is_sorted_non_decreasing(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i-1] > a[i]) return false;
    return true;
}

void bubble_sort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void selection_sort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int mn = i;
        for (int j = i + 1; j < n; ++j)
            if (a[j] < a[mn]) mn = j;
        if (mn != i) swap(a[i], a[mn]);
    }
}

void insertion_sort(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 1; i < n; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

template <class SortFn>
double bench_ms(SortFn sort_fn, const vector<int>& base) {
    vector<int> a = base;
    auto start = high_resolution_clock::now();
    sort_fn(a);
    auto end = high_resolution_clock::now();

    if (!is_sorted_non_decreasing(a)) {
        cerr << "ERROR: array is not sorted!\n";
    }

    return duration<double, milli>(end - start).count();
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> sizes = {100, 1000, 10000, 100000}; 
    uint64_t base_seed = 123456789ULL;

    cout << left << setw(14) << "Algorithm"
         << setw(10) << "N"
         << setw(12) << "Time (ms)"
         << "\n";
    cout << string(36, '-') << "\n";

    for (int n : sizes) {
        auto base = generate_array(n, base_seed + (uint64_t)n);

        double t_bubble = bench_ms(bubble_sort, base);
        cout << left << setw(14) << "Bubble" << setw(10) << n << setw(12) << fixed << setprecision(3) << t_bubble << "\n";

        double t_select = bench_ms(selection_sort, base);
        cout << left << setw(14) << "Selection" << setw(10) << n << setw(12) << fixed << setprecision(3) << t_select << "\n";

        double t_insert = bench_ms(insertion_sort, base);
        cout << left << setw(14) << "Insertion" << setw(10) << n << setw(12) << fixed << setprecision(3) << t_insert << "\n";

        cout << string(36, '-') << "\n";
    }

    return 0;
}