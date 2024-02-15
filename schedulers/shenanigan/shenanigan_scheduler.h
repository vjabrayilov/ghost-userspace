#ifndef GHOST_SCHEDULERS_SHENANIGAN_SHENANIGAN_SCHEDULER_H_
#define GHOST_SCHEDULERS_SHENANIGAN_SHENANIGAN_SCHEDULER_H_
#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "lib/agent.h"
#include "lib/scheduler.h"

namespace ghost {

struct ShenaniganTask : public Task<> {
  enum class RunState {
    kBlocked,
    kQueued,
    kRunnable,
    kOnCpu,
    kYielding,
  };

  ShenaniganTask(Gtid task_gtid, ghost_sw_info sw_info)
      : Task<>(task_gtid, sw_info) {}
  ~ShenaniganTask() override {}

  bool blocked() const { return run_state == RunState::kBlocked; }
  bool queued() const { return run_state == RunState::kQueued; }
  bool runnable() const { return run_state == RunState::kRunnable; }
  bool oncpu() const { return run_state == RunState::kOnCpu; }
  bool yielding() const { return run_state == RunState::kYielding; }

  static std::string_view RunStateToString(ShenaniganTask::RunState run_state) {
    switch (run_state) {
      case ShenaniganTask::RunState::kBlocked:
        return "Blocked";
      case ShenaniganTask::RunState::kQueued:
        return "Queued";
      case ShenaniganTask::RunState::kRunnable:
        return "Runnable";
      case ShenaniganTask::RunState::kOnCpu:
        return "OnCpu";
      case ShenaniganTask::RunState::kYielding:
        return "Yielding";
    }
    return {};
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  ShenaniganTask::RunState run_state) {
    return os << RunStateToString(run_state);
  }

  RunState run_state = RunState::kBlocked;
  Cpu cpu{Cpu::UninitializedType::kUninitialized};

  // Whether the last execution was preempted or not.
  bool preempted = false;
  bool prio_boost = false;
  pid_t vm_id = -1;
};

class ShenaniganScheduler : public BasicDispatchScheduler<ShenaniganTask> {
 public:
  explicit ShenaniganScheduler(
      Enclave* enclave, CpuList cpulist,
      std::shared_ptr<TaskAllocator<ShenaniganTask>> allocator,
      int32_t global_cpu, absl::Duration reallocation_interval);
  ~ShenaniganScheduler();

  void EnclaveReady();
  Channel& GetDefaultChannel() { return global_channel_; };

  // Handles task messages received from the kernel via shared memory queues.
  void TaskNew(ShenaniganTask* task, const Message& msg) final;
  void TaskRunnable(ShenaniganTask* task, const Message& msg) final;
  void TaskDeparted(ShenaniganTask* task, const Message& msg) final;
  void TaskDead(ShenaniganTask* task, const Message& msg) final;
  void TaskYield(ShenaniganTask* task, const Message& msg) final;
  void TaskBlocked(ShenaniganTask* task, const Message& msg) final;
  void TaskPreempted(ShenaniganTask* task, const Message& msg) final;

  bool Empty() { return num_tasks_ == 0; }
  
  // Removes 'task' from the runqueue.
  void RemoveFromRunqueue(ShenaniganTask* task);

  // Main scheduling function for the global agent.
  void GlobalSchedule(const StatusWord& agent_sw, BarrierToken agent_sw_last);

  int32_t GetGlobalCPUId() {
    return global_cpu_.load(std::memory_order_acquire);
  }

  void SetGlobalCPU(const Cpu& cpu) {
    global_cpu_core_ = cpu.core();
    global_cpu_.store(cpu.id(), std::memory_order_release);
  }

  // When a different scheduling class (e.g., CFS) has a task to run on the
  // global agent's CPU, the global agent calls this function to try to pick a
  // new CPU to move to and, if a new CPU is found, to initiate the handoff
  // process.
  // TODO: (vjabrayilov) Ideally this shouldn't be called frequently.
  bool PickNextGlobalCPU(BarrierToken agent_barrier, const Cpu& this_cpu);

  // Print debug details about the current tasks managed by the global agent,
  // CPU state, and runqueue stats.
  void DumpState(const Cpu& cpu, int flags);
  std::atomic<bool> debug_runqueue_ = false;

  static const int kDebugRunqueue = 1;

 private:
  struct CpuState {
    ShenaniganTask* current = nullptr;
    const Agent* agent = nullptr;
    absl::Time last_commit;
  } ABSL_CACHELINE_ALIGNED;

  // Updates the state of `task` to reflect that it is now running on `cpu`.
  // This method should be called after a transaction scheduling `task` onto
  // `cpu` succeeds.
  void TaskIsOnCpu(ShenaniganTask* task, const Cpu& cpu);

  // Marks a task as yielded.
  void Yield(ShenaniganTask* task);

  // Takes the task out of the yielding_tasks_ runqueue and puts it back into
  // the global runqueue.
  void Unyield(ShenaniganTask* task);

  // Adds a task (vCPU) to corresponding runqueue.
  void Enqueue(ShenaniganTask* task);

  // Removes and returns the task at the front of the runqueue.
  ShenaniganTask* Dequeue(pid_t vm_id);

  // Prints all tasks (includin tasks not running or on the runqueue) managed by
  // the global agent.
  void DumpAllTasks();

  // Returns 'true' if a CPU can be scheduled by ghOSt. Returns 'false'
  // otherwise, usually because a higher-priority scheduling class (e.g., CFS)
  // is currently using the CPU.
  bool Available(const Cpu& cpu);

  CpuState* cpu_state_of(const ShenaniganTask* task);

  CpuState* cpu_state(const Cpu& cpu) { return &cpu_states_[cpu.id()]; }

  CpuState cpu_states_[MAX_CPUS];

  int global_cpu_core_;
  std::atomic<int32_t> global_cpu_;
  LocalChannel global_channel_;
  int num_tasks_ = 0;
  absl::Duration reallocation_interval_;

  absl::flat_hash_map<pid_t, std::deque<ShenaniganTask*>> run_queues_;
  absl::flat_hash_map<pid_t, CpuList> cores_per_vm_;
  std::vector<ShenaniganTask*> yielding_tasks_;
};

// Initializes the task allocator and the Shinjuku scheduler.
std::unique_ptr<ShenaniganScheduler> SingleThreadShenaniganScheduler(
    Enclave* enclave, CpuList cpulist, int32_t global_cpu,
    absl::Duration reallocation_interval);

// Operates as the Global or Satellite agent depending on input from the
// global_scheduler->GetGlobalCPU callback.
class ShenaniganAgent : public LocalAgent {
 public:
  ShenaniganAgent(Enclave* enclave, Cpu cpu,
                  ShenaniganScheduler* global_scheduler)
      : LocalAgent(enclave, cpu), global_scheduler_(global_scheduler) {}
  void AgentThread() override;
  Scheduler* AgentScheduler() const override { return global_scheduler_; }

 private:
  ShenaniganScheduler* global_scheduler_;
};

class ShenaniganConfig : public AgentConfig {
 public:
  ShenaniganConfig() {}
  ShenaniganConfig(Topology* topology, CpuList cpulist, Cpu global_cpu,
                   absl::Duration reallocation_interval)
      : AgentConfig(topology, std::move(cpulist)),
        global_cpu_(global_cpu),
        reallocation_interval_(reallocation_interval) {}
  Cpu global_cpu_{Cpu::UninitializedType::kUninitialized};
  absl::Duration reallocation_interval_;
};

// A global agent scheduler that runs a single-threaded Shenanigan scheduler on
// the global_cpu_.
template <class EnclaveType>
class FullShenaniganAgent : public FullAgent<EnclaveType> {
 public:
  explicit FullShenaniganAgent(ShenaniganConfig config)
      : FullAgent<EnclaveType>(config) {
    global_scheduler_ = SingleThreadShenaniganScheduler(
        &this->enclave_, *this->enclave_.cpus(), config.global_cpu_.id(),
        config.reallocation_interval_);
    this->StartAgentTasks();
    this->enclave_.Ready();
  }

  ~FullShenaniganAgent() override {
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
    return std::make_unique<ShenaniganAgent>(&this->enclave_, cpu,
                                             global_scheduler_.get());
  }

  void RpcHandler(int64_t req, const AgentRpcArgs& args,
                  AgentRpcResponse& response) override {
    switch (req) {
      case ShenaniganScheduler::kDebugRunqueue:
        global_scheduler_->debug_runqueue_ = true;
        response.response_code = 0;
        return;
      default:
        response.response_code = -1;
        return;
    }
  }

 private:
  std::unique_ptr<ShenaniganScheduler> global_scheduler_;
};
}  // namespace ghost

#endif  // GHOST_SCHEDULERS_SHENANIGAN_SHENANIGAN_SCHEDULER_H_