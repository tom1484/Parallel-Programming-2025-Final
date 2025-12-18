#ifndef DSMC_PROFILER_H
#define DSMC_PROFILER_H

#include <cuda_runtime.h>

#include <map>
#include <string>

// =============================================================================
// Scoped CUDA Profiler
// Usage:
//   {
//       PROFILE(sort);
//       sort_particles();
//   }
//
// Enable/disable with -DDEBUG flag at compile time.
// Call Profiler::print_summary() at end of simulation to see results.
// =============================================================================

#ifdef DEBUG

class Profiler {
   public:
    struct TimingRecord {
        float total_ms = 0.0f;
        int count = 0;
        float min_ms = 1e9f;
        float max_ms = 0.0f;
    };

    // Singleton access
    static Profiler& instance();

    // Start timing a scope
    void start(const std::string& name);

    // Stop timing and record
    void stop(const std::string& name);

    // Print summary at end of simulation
    static void print_summary();

    // Dump summary to file
    static void dump_summary(const std::string& path);

    // Reset all records
    static void reset();

   private:
    Profiler() = default;
    ~Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    std::map<std::string, TimingRecord> records_;
    std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> active_timers_;
};

// RAII wrapper for automatic start/stop
class ScopedProfiler {
   public:
    explicit ScopedProfiler(const char* name);
    ~ScopedProfiler();

   private:
    std::string name_;
};

// Macro for easy scoped profiling
#define PROFILE(name) ScopedProfiler _profiler_##name(#name)

// Macro for manual start/stop (if needed)
#define PROFILE_START(name) Profiler::instance().start(#name)
#define PROFILE_STOP(name) Profiler::instance().stop(#name)

#else  // !DEBUG

// No-op implementations when DEBUG is not defined
class Profiler {
   public:
    static void print_summary() {}
    static void dump_summary(const std::string&) {}
    static void reset() {}
};

class ScopedProfiler {
   public:
    explicit ScopedProfiler(const char*) {}
};

#define PROFILE(name) ((void)0)
#define PROFILE_START(name) ((void)0)
#define PROFILE_STOP(name) ((void)0)

#endif  // DEBUG

#endif  // DSMC_PROFILER_H
