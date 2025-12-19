#include "profiler.h"

// #ifdef DEBUG

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

// Singleton instance
Profiler& Profiler::instance() {
    static Profiler inst;
    return inst;
}

// Start timing a scope
void Profiler::start(const std::string& name) {
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    active_timers_[name] = {start_event, stop_event};
}

// Stop timing and record
void Profiler::stop(const std::string& name) {
    auto it = active_timers_.find(name);
    if (it == active_timers_.end()) {
        fprintf(stderr, "Profiler: No active timer for '%s'\n", name.c_str());
        return;
    }

    cudaEvent_t start_event = it->second.first;
    cudaEvent_t stop_event = it->second.second;

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event, stop_event);

    // Update record
    TimingRecord& rec = records_[name];
    rec.total_ms += ms;
    rec.count++;
    rec.min_ms = (ms < rec.min_ms) ? ms : rec.min_ms;
    rec.max_ms = (ms > rec.max_ms) ? ms : rec.max_ms;

    // Cleanup
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    active_timers_.erase(it);
}

// Print summary at end of simulation
void Profiler::print_summary() {
    Profiler& p = instance();
    if (p.records_.empty()) {
        printf("\n[Profiler] No timing data recorded.\n");
        return;
    }

    printf("\n");
    printf("================================================================================\n");
    printf("                              PROFILER SUMMARY                                  \n");
    printf("================================================================================\n");
    printf("%-25s %10s %10s %10s %10s %10s\n", "Scope", "Calls", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
    printf("--------------------------------------------------------------------------------\n");

    float grand_total = 0.0f;
    std::vector<std::pair<std::string, TimingRecord>> sorted_records(p.records_.begin(), p.records_.end());

    // Sort by total time descending
    std::sort(sorted_records.begin(), sorted_records.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    for (const auto& kv : sorted_records) {
        const std::string& name = kv.first;
        const TimingRecord& rec = kv.second;
        float avg = rec.total_ms / rec.count;
        printf("%-25s %10d %10.3f %10.3f %10.3f %10.3f\n", name.c_str(), rec.count, rec.total_ms, avg, rec.min_ms,
               rec.max_ms);
        grand_total += rec.total_ms;
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("%-25s %10s %10.3f\n", "TOTAL", "", grand_total);
    printf("================================================================================\n\n");
}

// Dump summary to file
void Profiler::dump_summary(const std::string& path) {
    Profiler& p = instance();
    
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Profiler: Failed to open file '%s' for writing\n", path.c_str());
        return;
    }

    if (p.records_.empty()) {
        fprintf(f, "[Profiler] No timing data recorded.\n");
        fclose(f);
        return;
    }

    fprintf(f, "================================================================================\n");
    fprintf(f, "                              PROFILER SUMMARY                                  \n");
    fprintf(f, "================================================================================\n");
    fprintf(f, "%-25s %10s %10s %10s %10s %10s\n", "Scope", "Calls", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
    fprintf(f, "--------------------------------------------------------------------------------\n");

    float grand_total = 0.0f;
    std::vector<std::pair<std::string, TimingRecord>> sorted_records(p.records_.begin(), p.records_.end());

    // Sort by total time descending
    std::sort(sorted_records.begin(), sorted_records.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

    for (const auto& kv : sorted_records) {
        const std::string& name = kv.first;
        const TimingRecord& rec = kv.second;
        float avg = rec.total_ms / rec.count;
        fprintf(f, "%-25s %10d %10.3f %10.3f %10.3f %10.3f\n", name.c_str(), rec.count, rec.total_ms, avg, rec.min_ms,
               rec.max_ms);
        grand_total += rec.total_ms;
    }

    fprintf(f, "--------------------------------------------------------------------------------\n");
    fprintf(f, "%-25s %10s %10.3f\n", "TOTAL", "", grand_total);
    fprintf(f, "================================================================================\n");

    fclose(f);
}

// Reset all records
void Profiler::reset() {
    Profiler& p = instance();
    p.records_.clear();
    p.active_timers_.clear();
}

// ScopedProfiler implementation
ScopedProfiler::ScopedProfiler(const char* name) : name_(name) {
    Profiler::instance().start(name_);
}

ScopedProfiler::~ScopedProfiler() {
    Profiler::instance().stop(name_);
}

// #endif  // DEBUG
