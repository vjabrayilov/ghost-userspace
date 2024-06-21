// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef GHOST_SCHEDULERS_CAF_SCHEDULER_H
#define GHOST_SCHEDULERS_CAF_SCHEDULER_H

#include <boost/histogram.hpp>
#include <map>
#include <memory>
#include <unordered_map>

#include "absl/time/time.h"
#include "lib/agent.h"
#include "lib/scheduler.h"

namespace ghost {

// Store information about a scheduled task.
struct CafTask : public Task<> {
  enum class RunState {
    kBlocked,
    kQueued,
    kRunnable,
    kOnCpu,
    kYielding,
  };

  CafTask(Gtid fifo_task_gtid, ghost_sw_info sw_info)
      : Task<>(fifo_task_gtid, sw_info) {}
  ~CafTask() override {}

  bool blocked() const { return run_state == RunState::kBlocked; }
  bool queued() const { return run_state == RunState::kQueued; }
  bool runnable() const { return run_state == RunState::kRunnable; }
  bool oncpu() const { return run_state == RunState::kOnCpu; }
  bool yielding() const { return run_state == RunState::kYielding; }

  static std::string_view RunStateToString(CafTask::RunState run_state) {
    switch (run_state) {
      case CafTask::RunState::kBlocked:
        return "Blocked";
      case CafTask::RunState::kQueued:
        return "Queued";
      case CafTask::RunState::kRunnable:
        return "Runnable";
      case CafTask::RunState::kOnCpu:
        return "OnCpu";
      case CafTask::RunState::kYielding:
        return "Yielding";
    }
    return "Unknown";
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  CafTask::RunState run_state) {
    return os << RunStateToString(run_state);
  }

  RunState run_state = RunState::kBlocked;
  Cpu cpu{Cpu::UninitializedType::kUninitialized};

  // Whether the last execution was preempted or not.
  bool preempted = false;
  bool prio_boost = false;
  pid_t vm_id = -1;
};
class DynamicLatencyRecorder {
 public:
  DynamicLatencyRecorder() {
    // Calculate the number of bins for two-digit precision
    // Logarithmic scale: base 10
    int minLog = std::log10(1);
    int maxLog = std::log10(100000000);      // 100 seconds in microseconds
    precision = (maxLog - minLog + 1) * 10;  // 10 bins per power of 10
    histogram.resize(precision, 0);
    maxRange = 100000000;  // 100 seconds in microseconds
  }

  void record(uint64_t value) {
    if (value < 1 || value > maxRange) {
      std::cerr << "Value out of range." << std::endl;
      return;
    }

    int index = std::log10(value) * 10;
    histogram[index]++;
  }

  void printPercentiles() {
    std::vector<double> percentiles = {0.50, 0.99, 0.999};
    uint64_t total =
        std::accumulate(histogram.begin(), histogram.end(), uint64_t(0));

    for (double percentile : percentiles) {
      uint64_t count = 0;
      uint64_t targetCount = static_cast<uint64_t>(percentile * total);
      int percentileIndex = 0;

      for (; percentileIndex < precision; ++percentileIndex) {
        count += histogram[percentileIndex];
        if (count >= targetCount) break;
      }

      double lower_bound =
          std::pow(10, static_cast<double>(percentileIndex) / 10);
      std::cout << "Percentile " << (percentile * 100) << ": " << lower_bound
                << " microseconds" << std::endl;
    }
  }

  void clear() { std::fill(histogram.begin(), histogram.end(), 0); }

 private:
  uint64_t maxRange;
  int precision;
  std::vector<int> histogram;
};

class CafScheduler : public BasicDispatchScheduler<CafTask> {
 public:
  CafScheduler(Enclave* enclave, CpuList cpulist,
               std::shared_ptr<TaskAllocator<CafTask>> allocator,
               int32_t global_cpu, absl::Duration preemption_time_slice,
               absl::Duration reallocation_interval);
  ~CafScheduler();

  void EnclaveReady();
  Channel& GetDefaultChannel() { return global_channel_; };

  // Handles task messages received from the kernel via shared memory queues.
  void TaskNew(CafTask* task, const Message& msg);
  void TaskRunnable(CafTask* task, const Message& msg);
  void TaskDeparted(CafTask* task, const Message& msg);
  void TaskDead(CafTask* task, const Message& msg);
  void TaskYield(CafTask* task, const Message& msg);
  void TaskBlocked(CafTask* task, const Message& msg);
  void TaskPreempted(CafTask* task, const Message& msg);

  // Handles cpu "not idle" message. Currently a nop.
  void CpuNotIdle(const Message& msg);

  // Handles cpu "timer expired" messages. Currently a nop.
  void CpuTimerExpired(const Message& msg);

  bool Empty() { return num_tasks_ == 0; }

  // Removes 'task' from the runqueue.
  void RemoveFromRunqueue(CafTask* task);

  // Main scheduling function for the global agent.
  void GlobalSchedule(const StatusWord& agent_sw, BarrierToken agent_sw_last);

  int32_t GetGlobalCPUId() {
    return global_cpu_.load(std::memory_order_acquire);
  }

  void SetGlobalCPU(const Cpu& cpu) {
    global_cpu_core_ = cpu.core();
    global_cpu_.store(cpu.id(), std::memory_order_release);
  }

  void ReallocateCores(bool forcefully);

  // When a different scheduling class (e.g., CFS) has a task to run on the
  // global agent's CPU, the global agent calls this function to try to pick a
  // new CPU to move to and, if a new CPU is found, to initiate the handoff
  // process.
  bool PickNextGlobalCPU(BarrierToken agent_barrier, const Cpu& this_cpu);

  // Print debug details about the current tasks managed by the global agent,
  // CPU state, and runqueue stats.
  void DumpState(const Cpu& cpu, int flags);
  std::atomic<bool> debug_runqueue_ = false;

  static const int kDebugRunqueue = 1;

 private:
  struct CpuState {
    CafTask* current = nullptr;
    const Agent* agent = nullptr;
    absl::Time last_commit;
    pid_t vm_id = -1;
  } ABSL_CACHELINE_ALIGNED;

  // Updates the state of `task` to reflect that it is now running on `cpu`.
  // This method should be called after a transaction scheduling `task` onto
  // `cpu` succeeds.
  void TaskOnCpu(CafTask* task, const Cpu& cpu);

  // Marks a task as yielded.
  void Yield(CafTask* task);
  // Takes the task out of the yielding_tasks_ runqueue and puts it back into
  // the global runqueue.
  void Unyield(CafTask* task);

  // Adds a task to the FIFO runqueue.
  void Enqueue(CafTask* task);

  // Removes and returns the task at the front of the runqueue.
  CafTask* Dequeue(pid_t vm_id);

  // Prints all tasks (includin tasks not running or on the runqueue) managed by
  // the global agent.
  void DumpAllTasks();

  // Returns 'true' if a CPU can be scheduled by ghOSt. Returns 'false'
  // otherwise, usually because a higher-priority scheduling class (e.g., CFS)
  // is currently using the CPU.
  bool Available(const Cpu& cpu);

  CpuState* cpu_state_of(const CafTask* task);

  CpuState* cpu_state(const Cpu& cpu) { return &cpu_states_[cpu.id()]; }

  size_t RunqueueSize(pid_t vm_id) const {
    return vm_run_queues_.at(vm_id).size();
  }

  bool RunqueueEmpty(pid_t vm_id) const { return RunqueueSize(vm_id) == 0; }

  CpuState cpu_states_[MAX_CPUS];

  int global_cpu_core_;
  std::atomic<int32_t> global_cpu_;
  LocalChannel global_channel_;
  int num_tasks_ = 0;

  const absl::Duration preemption_time_slice_;

  std::map<pid_t, std::deque<CafTask*>> vm_run_queues_;
  std::vector<CafTask*> yielding_tasks_;

  absl::Time schedule_timer_start_;
  absl::Duration schedule_durations_;
  uint64_t iterations_ = 0;
  absl::Time last_blocked = absl::InfinitePast();
  std::vector<pid_t> new_vm_joined_ = {};
  std::unordered_map<pid_t, int64_t> vm_running_vcpus_;
  const absl::Duration reallocation_interval_;
  absl::Time last_reallocation_time_ = absl::InfinitePast();
};

// Initializes the task allocator and the FIFO scheduler.
std::unique_ptr<CafScheduler> SingleThreadCafScheduler(
    Enclave* enclave, CpuList cpulist, int32_t global_cpu,
    absl::Duration preemption_time_slice, absl::Duration reallocation_interval);

// Operates as the Global or Satellite agent depending on input from the
// global_scheduler->GetGlobalCPU callback.
class CafAgent : public LocalAgent {
 public:
  CafAgent(Enclave* enclave, Cpu cpu, CafScheduler* global_scheduler)
      : LocalAgent(enclave, cpu), global_scheduler_(global_scheduler) {}

  void AgentThread() override;
  Scheduler* AgentScheduler() const override { return global_scheduler_; }

 private:
  CafScheduler* global_scheduler_;
};

class CafConfig : public AgentConfig {
 public:
  CafConfig() {}
  CafConfig(Topology* topology, CpuList cpulist, Cpu global_cpu,
            absl::Duration preemption_time_slice,
            absl::Duration reallocation_interval)
      : AgentConfig(topology, std::move(cpulist)),
        global_cpu_(global_cpu),
        preemption_time_slice_(preemption_time_slice),
        reallocation_interval_(reallocation_interval) {}

  Cpu global_cpu_{Cpu::UninitializedType::kUninitialized};
  absl::Duration preemption_time_slice_ = absl::Microseconds(50);
  absl::Duration reallocation_interval_ = absl::Microseconds(100);
};

// A global agent scheduler. It runs a single-threaded FIFO scheduler on the
// global_cpu.
template <class EnclaveType>
class FullCafAgent : public FullAgent<EnclaveType> {
 public:
  explicit FullCafAgent(CafConfig config) : FullAgent<EnclaveType>(config) {
    global_scheduler_ = SingleThreadCafScheduler(
        &this->enclave_, *this->enclave_.cpus(), config.global_cpu_.id(),
        config.preemption_time_slice_, config.reallocation_interval_);
    this->StartAgentTasks();
    this->enclave_.Ready();
  }

  ~FullCafAgent() override {
    // Terminate global agent before satellites to avoid a false negative error
    // from ghost_run(). e.g. when the global agent tries to schedule on a CPU
    // without an active satellite agent.
    auto global_cpuid = global_scheduler_->GetGlobalCPUId();

    if (this->agents_.front()->cpu().id() != global_cpuid) {
      // Bring the current globalcpu agent to the front.
      for (auto it = this->agents_.begin(); it != this->agents_.end(); it++) {
        if (((*it)->cpu().id() == global_cpuid)) {
          auto d = std::distance(this->agents_.begin(), it);
          std::iter_swap(this->agents_.begin(), this->agents_.begin() + d);
          break;
        }
      }
    }

    CHECK_EQ(this->agents_.front()->cpu().id(), global_cpuid);

    this->TerminateAgentTasks();
  }

  std::unique_ptr<Agent> MakeAgent(const Cpu& cpu) override {
    return std::make_unique<CafAgent>(&this->enclave_, cpu,
                                      global_scheduler_.get());
  }

  void RpcHandler(int64_t req, const AgentRpcArgs& args,
                  AgentRpcResponse& response) override {
    switch (req) {
      case CafScheduler::kDebugRunqueue:
        global_scheduler_->debug_runqueue_ = true;
        response.response_code = 0;
        return;
      default:
        response.response_code = -1;
        return;
    }
  }

 private:
  std::unique_ptr<CafScheduler> global_scheduler_;
};

}  // namespace ghost

#endif
