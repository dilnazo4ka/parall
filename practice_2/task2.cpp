
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace chrono;

// -------------------- Utils --------------------

static vector<int> generate_array(int n, uint64_t seed) {
    mt19937_64 gen(seed);
    uniform_int_distribution<int> dist(0, 100000);
    vector<int> a(n);
    for (int &x : a) x = dist(gen);
    return a;
}

static bool is_sorted_non_decreasing(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i)
        if (a[i - 1] > a[i]) return false;
    return true;
}

template <class F>
static double time_ms(F&& fn) {
    auto t0 = high_resolution_clock::now();
    fn();
    auto t1 = high_resolution_clock::now();
    return duration<double, milli>(t1 - t0).count();
}

// -------------------- Sequential sorts --------------------

static void bubble_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

static void selection_sort_seq(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int mn = i;
        for (int j = i + 1; j < n; ++j)
            if (a[j] < a[mn]) mn = j;
        if (mn != i) swap(a[i], a[mn]);
    }
}

static void insertion_sort_seq(vector<int>& a) {
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


// Bubble-like parallel: odd-even transposition sort (canonical OpenMP loop)
static void bubble_sort_omp(vector<int>& a) {
    int n = (int)a.size();
    if (n < 2) return;

    int limit = n - 1; // ensures canonical predicate: j < limit

    for (int phase = 0; phase < n; ++phase) {
        int start = (phase & 1) ? 1 : 0;

        #pragma omp parallel for schedule(static)
        for (int j = start; j < limit; j += 2) {
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
        }
    }
}

// Selection: parallelize scanning j-loop for minimum
static void selection_sort_omp(vector<int>& a) {
    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int global_min_idx = i;

        #pragma omp parallel
        {
            int local_min_idx = i;

            #pragma omp for nowait schedule(static)
            for (int j = i + 1; j < n; ++j) {
                if (a[j] < a[local_min_idx]) local_min_idx = j;
            }

            #pragma omp critical
            {
                if (a[local_min_idx] < a[global_min_idx])
                    global_min_idx = local_min_idx;
            }
        }

        if (global_min_idx != i) swap(a[i], a[global_min_idx]);
    }
}

// Insertion: block insertion sort + parallel merges
static void insertion_sort_range(vector<int>& a, int l, int r) { // [l, r)
    for (int i = l + 1; i < r; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= l && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

static vector<int> merge_two(const vector<int>& L, const vector<int>& R) {
    vector<int> out(L.size() + R.size());
    size_t i = 0, j = 0, k = 0;
    while (i < L.size() && j < R.size()) {
        if (L[i] <= R[j]) out[k++] = L[i++];
        else out[k++] = R[j++];
    }
    while (i < L.size()) out[k++] = L[i++];
    while (j < R.size()) out[k++] = R[j++];
    return out;
}

static void insertion_sort_omp(vector<int>& a) {
    int n = (int)a.size();
    if (n < 2) return;

    int T = omp_get_max_threads();
    if (T < 1) T = 1;

    vector<pair<int,int>> blocks;
    blocks.reserve(T);
    for (int t = 0; t < T; ++t) {
        int l = (long long)t * n / T;
        int r = (long long)(t + 1) * n / T;
        if (l < r) blocks.push_back({l, r});
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < (int)blocks.size(); ++b) {
        insertion_sort_range(a, blocks[b].first, blocks[b].second);
    }

    vector<vector<int>> runs;
    runs.reserve(blocks.size());
    for (auto [l, r] : blocks)
        runs.emplace_back(a.begin() + l, a.begin() + r);

    while (runs.size() > 1) {
        int m = (int)runs.size();
        vector<vector<int>> next((m + 1) / 2);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i += 2) {
            int idx = i / 2;
            if (i + 1 < m) next[idx] = merge_two(runs[i], runs[i + 1]);
            else next[idx] = std::move(runs[i]);
        }

        runs = std::move(next);
    }

    if (!runs.empty()) a = std::move(runs[0]);
}

// -------------------- Benchmark --------------------

static void bench_algo(const string& name,
                       void(*seq_fn)(vector<int>&),
                       void(*omp_fn)(vector<int>&),
                       const vector<int>& base) {
    vector<int> a1 = base;
    vector<int> a2 = base;

    double t_seq = time_ms([&]{ seq_fn(a1); });
    double t_omp = time_ms([&]{ omp_fn(a2); });

    if (!is_sorted_non_decreasing(a1) || !is_sorted_non_decreasing(a2) || a1 != a2) {
        cerr << "ERROR: " << name << " mismatch / not sorted\n";
    }

    cout << left << setw(12) << name
         << right << setw(10) << base.size()
         << setw(12) << fixed << setprecision(3) << t_seq
         << setw(12) << fixed << setprecision(3) << t_omp
         << setw(10) << fixed << setprecision(2) << (t_seq / max(1e-12, t_omp))
         << "\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> sizes = {100, 1000, 10000, 100000};
    uint64_t seed0 = 123456789ULL;

    int maxT = omp_get_max_threads();
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    thread_counts.erase(
        remove_if(thread_counts.begin(), thread_counts.end(),
                  [&](int t){ return t > maxT; }),
        thread_counts.end()
    );
    if (thread_counts.empty()) thread_counts = {maxT};

    for (int th : thread_counts) {
        omp_set_num_threads(th);

        cout << "\n=== OpenMP threads: " << th << " (max: " << maxT << ") ===\n";
        cout << left << setw(12) << "Algorithm"
             << right << setw(10) << "N"
             << setw(12) << "Seq ms"
             << setw(12) << "OMP ms"
             << setw(10) << "Speedup"
             << "\n";
        cout << string(56, '-') << "\n";

        for (int n : sizes) {
            auto base = generate_array(n, seed0 + (uint64_t)n);

            bench_algo("Bubble",    bubble_sort_seq,    bubble_sort_omp,    base);
            bench_algo("Selection", selection_sort_seq, selection_sort_omp, base);
            bench_algo("Insertion", insertion_sort_seq, insertion_sort_omp, base);

            cout << string(56, '-') << "\n";
        }
    }

    return 0;
}