// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "schedulers/core_allocator_fifo/caf_scheduler.h"

#include <unistd.h>

#include <memory>

#include "absl/strings/str_format.h"

namespace ghost {

void CafScheduler::CpuNotIdle(const Message& msg) { CHECK(0); }

void CafScheduler::CpuTimerExpired(const Message& msg) { CHECK(0); }

CafScheduler::CafScheduler(Enclave* enclave, CpuList cpulist,
                           std::shared_ptr<TaskAllocator<CafTask>> allocator,
                           int32_t global_cpu,
                           absl::Duration preemption_time_slice,
                           absl::Duration reallocation_interval)
    : BasicDispatchScheduler(enclave, std::move(cpulist), std::move(allocator)),
      global_cpu_(global_cpu),
      global_channel_(GHOST_MAX_QUEUE_ELEMS, /*node=*/0),
      preemption_time_slice_(preemption_time_slice),
      reallocation_interval_(reallocation_interval) {
  if (!cpus().IsSet(global_cpu_)) {
    Cpu c = cpus().Front();
    CHECK(c.valid());
    global_cpu_ = c.id();
  }
}

CafScheduler::~CafScheduler() {}

void CafScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    cs->agent = enclave()->GetAgent(cpu);
    CHECK_NE(cs->agent, nullptr);
  }
}

bool CafScheduler::Available(const Cpu& cpu) {
  CpuState* cs = cpu_state(cpu);

  if (cs->agent) return cs->agent->cpu_avail();

  return false;
}

void CafScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state       rq_pos  P\n");
  allocator()->ForEachTask([](Gtid gtid, const CafTask* task) {
    absl::FPrintF(stderr, "%-12s%-12s%d\n", gtid.describe(),
                  CafTask::RunStateToString(task->run_state),
                  task->cpu.valid() ? task->cpu.id() : -1);
    return true;
  });
}

void CafScheduler::DumpState(const Cpu& agent_cpu, int flags) {
  if (flags & kDumpAllTasks) {
    DumpAllTasks();
  }
  pid_t vm_id = cpu_state(agent_cpu)->vm_id;
  if (!(flags & kDumpStateEmptyRQ) && RunqueueEmpty(vm_id)) {
    return;
  }

  fprintf(stderr, "SchedState: ");
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    fprintf(stderr, "%d:", cpu.id());
    if (!cs->current) {
      fprintf(stderr, "none ");
    } else {
      Gtid gtid = cs->current->gtid;
      absl::FPrintF(stderr, "%s ", gtid.describe());
    }
  }
  fprintf(stderr, " vm(%d): rq_l=%ld", vm_id, RunqueueSize(vm_id));
  fprintf(stderr, "\n");
}

CafScheduler::CpuState* CafScheduler::cpu_state_of(const CafTask* task) {
  CHECK(task->cpu.valid());
  CHECK(task->oncpu());
  CpuState* cs = cpu_state(task->cpu);
  CHECK(task == cs->current);
  return cs;
}

void CafScheduler::TaskNew(CafTask* task, const Message& msg) {
  int ret;
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());
  DLOG(INFO) << absl::StrFormat("TaskNew: %s", task->gtid.describe());

  task->seqnum = msg.seqnum();
  task->run_state = CafTask::RunState::kBlocked;

  const Gtid gtid(payload->gtid);
  task->vm_id = gtid.tgid();
  if (vm_run_queues_.find(task->vm_id) == vm_run_queues_.end()) {
    new_vm_joined_.push_back(task->vm_id);
  }
  if (payload->runnable) {
    task->run_state = CafTask::RunState::kRunnable;
    Enqueue(task);
  }

  num_tasks_++;
}

void CafScheduler::TaskRunnable(CafTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());
  DLOG(INFO) << absl::StrFormat("TaskRunnable: %s", task->gtid.describe());
  CHECK(task->blocked());

  task->run_state = CafTask::RunState::kRunnable;
  task->prio_boost = !payload->deferrable;
  Enqueue(task);
  if (!new_vm_joined_.empty()) {
    ReallocateCores(true);
    new_vm_joined_.pop_back();
  }
}

void CafScheduler::TaskDeparted(CafTask* task, const Message& msg) {
  DLOG(INFO) << absl::StrFormat("TaskDeparted: %s", task->gtid.describe());
  if (task->yielding()) {
    Unyield(task);
  }

  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else if (task->queued()) {
    RemoveFromRunqueue(task);
  } else {
    CHECK(task->blocked());
  }

  allocator()->FreeTask(task);
  num_tasks_--;
}

void CafScheduler::TaskDead(CafTask* task, const Message& msg) {
  DLOG(INFO) << absl::StrFormat("TaskDead: %s", task->gtid.describe());
  CHECK_EQ(task->run_state, CafTask::RunState::kBlocked);
  allocator()->FreeTask(task);
  num_tasks_--;
}

void CafScheduler::TaskBlocked(CafTask* task, const Message& msg) {
  DLOG(INFO) << absl::StrFormat("TaskBlocked: %s", task->gtid.describe());
  last_blocked = MonotonicNow();
  // auto start = MonotonicNow();
  // auto end = MonotonicNow();
  // duration in microseconds
  // auto duration = absl::ToInt64Microseconds(end - start);
  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);

    vm_running_vcpus_[task->vm_id]--;
    CHECK_GE(vm_running_vcpus_[task->vm_id], 0);

    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(task->queued());
    RemoveFromRunqueue(task);
  }

  task->run_state = CafTask::RunState::kBlocked;
}

void CafScheduler::TaskPreempted(CafTask* task, const Message& msg) {
  DLOG(INFO) << absl::StrFormat("TaskPreempted: %s", task->gtid.describe());
  task->preempted = true;

  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
    task->run_state = CafTask::RunState::kRunnable;
    Enqueue(task);
    vm_running_vcpus_[task->vm_id]--;
    CHECK_GE(vm_running_vcpus_[task->vm_id], 0) << " vm_id: " << task->vm_id;
  } else {
    CHECK(task->queued());
  }
}

void CafScheduler::TaskYield(CafTask* task, const Message& msg) {
  DLOG(INFO) << absl::StrFormat("TaskYield: %s", task->gtid.describe());
  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    vm_running_vcpus_[task->vm_id]--;
    CHECK_GE(vm_running_vcpus_[task->vm_id], 0);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
    Yield(task);
  } else {
    CHECK(task->queued());
  }
}

void CafScheduler::Yield(CafTask* task) {
  // An oncpu() task can do a sched_yield() and get here via TaskYield().
  // We may also get here if the scheduler wants to inhibit a task from being
  // picked in the current scheduling round (see GlobalSchedule()).
  CHECK(task->oncpu() || task->runnable());
  task->run_state = CafTask::RunState::kYielding;
  yielding_tasks_.emplace_back(task);
}

void CafScheduler::Unyield(CafTask* task) {
  CHECK(task->yielding());

  auto it = std::find(yielding_tasks_.begin(), yielding_tasks_.end(), task);
  CHECK(it != yielding_tasks_.end());
  yielding_tasks_.erase(it);

  task->run_state = CafTask::RunState::kRunnable;
  Enqueue(task);
}

void CafScheduler::Enqueue(CafTask* task) {
  CHECK_EQ(task->run_state, CafTask::RunState::kRunnable);
  task->run_state = CafTask::RunState::kQueued;
  if (task->prio_boost || task->preempted) {
    vm_run_queues_[task->vm_id].push_front(task);
  } else {
    vm_run_queues_[task->vm_id].push_back(task);
  }
}

CafTask* CafScheduler::Dequeue(pid_t vm_id) {
  if (vm_id == -1 || RunqueueEmpty(vm_id)) {
    return nullptr;
  }

  CafTask* task = vm_run_queues_[vm_id].front();
  CHECK_EQ(task->run_state, CafTask::RunState::kQueued);
  task->run_state = CafTask::RunState::kRunnable;
  vm_run_queues_[vm_id].pop_front();

  return task;
}

void CafScheduler::RemoveFromRunqueue(CafTask* task) {
  CHECK(task->queued());

  auto& run_queue_ = vm_run_queues_[task->vm_id];
  for (int pos = run_queue_.size() - 1; pos >= 0; pos--) {
    // The [] operator for 'std::deque' is constant time
    if (run_queue_[pos] == task) {
      // Caller is responsible for updating 'run_state' if task is
      // no longer runnable.
      task->run_state = CafTask::RunState::kRunnable;
      run_queue_.erase(run_queue_.cbegin() + pos);
      return;
    }
  }

  // This state is unreachable because the task is queued.
  CHECK(false);
}

void CafScheduler::TaskOnCpu(CafTask* task, const Cpu& cpu) {
  CpuState* cs = cpu_state(cpu);
  CHECK_EQ(task, cs->current);
  CHECK_EQ(task->vm_id, cs->vm_id);

  GHOST_DPRINT(3, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = CafTask::RunState::kOnCpu;
  task->cpu = cpu;
  task->preempted = false;
  task->prio_boost = false;
}

void CafScheduler::GlobalSchedule(const StatusWord& agent_sw,
                                  BarrierToken agent_sw_last) {
  const int global_cpu_id = GetGlobalCPUId();
  CpuList available = topology()->EmptyCpuList();
  CpuList assigned = topology()->EmptyCpuList();

  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);

    if (cpu.id() == global_cpu_id) {
      CHECK_EQ(cs->current, nullptr);
      continue;
    }

    if (!Available(cpu)) {
      // This CPU is running a higher priority sched class, such as CFS.
      continue;
    }
    if (cs->current &&
        (MonotonicNow() - cs->last_commit) < preemption_time_slice_) {
      // This CPU is currently running a task, so do not schedule a different
      // task on it.
      continue;
    }
    // No task is running on this CPU, so designate this CPU as available.
    available.Set(cpu);
  }

  while (!available.Empty()) {
    // Assign `next` to run on the CPU at the front of `available`.
    const Cpu& next_cpu = available.Front();
    CpuState* cs = cpu_state(next_cpu);

    CafTask* next = Dequeue(cs->vm_id);
    if (!next) {
      break;
    }

    // If `next->status_word.on_cpu()` is true, then `next` was previously
    // preempted by this scheduler but hasn't been moved off the CPU it was
    // previously running on yet.
    //
    // If `next->seqnum != next->status_word.barrier()` is true, then there are
    // pending messages for `next` that we have not read yet. Thus, do not
    // schedule `next` since we need to read the messages. We will schedule
    // `next` in a future iteration of the global scheduling loop.
    if (next->status_word.on_cpu() ||
        next->seqnum != next->status_word.barrier()) {
      Yield(next);
      continue;
    }

    if (cs->current) {
      cs->current->run_state = CafTask::RunState::kRunnable;
      Enqueue(cs->current);
      vm_running_vcpus_[cs->current->vm_id]--;
      CHECK_GE(vm_running_vcpus_[cs->current->vm_id], 0);
    }
    cs->current = next;

    available.Clear(next_cpu);
    assigned.Set(next_cpu);

    CHECK(cs->vm_id == next->vm_id);
    CHECK(cs->vm_id != GetGlobalCPUId());

    vm_running_vcpus_[cs->current->vm_id]++;
    CHECK_LE(vm_running_vcpus_[cs->current->vm_id], 4);

    RunRequest* req = enclave()->GetRunRequest(next_cpu);
    req->Open({.target = next->gtid,
               .target_barrier = next->seqnum,
               // No need to set `agent_barrier` because the agent barrier is
               // not checked when a global agent is scheduling a CPU other than
               // the one that the global agent is currently running on.
               .commit_flags = COMMIT_AT_TXN_COMMIT,
               .sync_group_owner = next_cpu.id(),
               .allow_txn_target_on_cpu = true});
  }

  // Commit on all CPUs with open transactions.
  if (!assigned.Empty()) {
    enclave()->CommitRunRequests(assigned);
    absl::Time now = MonotonicNow();
    for (const Cpu& cpu : assigned) {
      cpu_state(cpu)->last_commit = now;
    }
  }
  for (const Cpu& next_cpu : assigned) {
    CpuState* cs = cpu_state(next_cpu);
    RunRequest* req = enclave()->GetRunRequest(next_cpu);
    if (req->succeeded()) {
      // The transaction succeeded and `next` is running on `next_cpu`.
      TaskOnCpu(cs->current, next_cpu);
    } else {
      GHOST_DPRINT(3, stderr,
                   "CafSchedulCafSchedulee: commit failed (state=%d)",
                   req->state());

      // The transaction commit failed so push `next` to the front of runqueue.
      cs->current->prio_boost = true;
      Enqueue(cs->current);
      vm_running_vcpus_[cs->current->vm_id]--;
      CHECK_GE(vm_running_vcpus_[cs->current->vm_id], 0);
      // The task failed to run on `next_cpu`, so clear out `cs->current`.
      cs->current = nullptr;
    }
  }

  // Yielding tasks are moved back to the runqueue having skipped one round
  // of scheduling decisions.
  if (!yielding_tasks_.empty()) {
    for (CafTask* t : yielding_tasks_) {
      CHECK_EQ(t->run_state, CafTask::RunState::kYielding);
      t->run_state = CafTask::RunState::kRunnable;
      Enqueue(t);
    }
    yielding_tasks_.clear();
  }
}

void CafScheduler::ReallocateCores(bool forcefully) {
  // if no new vm joined or reallocation_interval not reached skip reallocating
  if (MonotonicNow() - last_reallocation_time_ < reallocation_interval_) {
    if (!forcefully) {
      return;
    }
  }

  last_reallocation_time_ = MonotonicNow();
  auto pcpu_list = cpus().ToVector();
  auto global_cpu_id = GetGlobalCPUId();
  // remove global cpu from the pcpu_list
  pcpu_list.erase(std::remove_if(pcpu_list.begin(), pcpu_list.end(),
                                 [global_cpu_id](const Cpu& cpu) {
                                   return cpu.id() == global_cpu_id;
                                 }),
                  pcpu_list.end());
  auto pcpu_count = pcpu_list.size();

  // Implemenet core reallocation
  // each vm get number of cores proportional to their queue size
  size_t total_runnable_vcpus = 0;
  for (const auto& [vm_id, rq] : vm_run_queues_) {
    total_runnable_vcpus += RunqueueSize(vm_id);
  }

  if (total_runnable_vcpus < pcpu_count) {
    size_t i = 0;
    for (const auto& [vm_id, rq] : vm_run_queues_) {
      size_t per_vm_quota = RunqueueSize(vm_id);
      for (; i < per_vm_quota; i++) {
        CHECK(pcpu_list[i].id() != global_cpu_id);
        cpu_state(pcpu_list[i])->vm_id = vm_id;
      }

      per_vm_quota = pcpu_count;  /// vm_run_queues_.size();
      CHECK_NE(per_vm_quota, 0);
    }
  } else {
    DLOG(WARNING) << "Assigning cores proportional to the # of runnable vCPUs.";
    for (const auto& [vm_id, rq] : vm_run_queues_) {
      auto per_vm_quota =
          std::lround(pcpu_count * static_cast<double>(RunqueueSize(vm_id)) /
                      total_runnable_vcpus);
      CHECK_LE(per_vm_quota, pcpu_list.size());
      for (size_t j = 0; j < per_vm_quota; j++) {
        cpu_state(pcpu_list.back())->vm_id = vm_id;
        pcpu_list.pop_back();
        // cpu_state(pcpu_list[i + j])->vm_id = vm_id;
      }

      if (per_vm_quota == 0) {
        continue;
      }
      CHECK_NE(per_vm_quota, 0);
    }
  }
}

bool CafScheduler::PickNextGlobalCPU(BarrierToken agent_barrier,
                                     const Cpu& this_cpu) {
  Cpu target(Cpu::UninitializedType::kUninitialized);
  Cpu global_cpu = topology()->cpu(GetGlobalCPUId());
  int numa_node = global_cpu.numa_node();

  // Let's make sure we do some useful work before moving to another CPU.
  if (iterations_ & 0xff) {
    return false;
  }

  for (const Cpu& cpu : global_cpu.siblings()) {
    if (cpu.id() == global_cpu.id()) continue;

    if (Available(cpu)) {
      target = cpu;
      goto found;
    }
  }

  for (const Cpu& cpu : global_cpu.l3_siblings()) {
    if (cpu.id() == global_cpu.id()) continue;

    if (Available(cpu)) {
      target = cpu;
      goto found;
    }
  }

again:
  for (const Cpu& cpu : cpus()) {
    if (cpu.id() == global_cpu.id()) continue;

    if (numa_node >= 0 && cpu.numa_node() != numa_node) continue;

    if (Available(cpu)) {
      target = cpu;
      goto found;
    }
  }

  if (numa_node >= 0) {
    numa_node = -1;
    goto again;
  }

found:
  if (!target.valid()) return false;

  CHECK(target != this_cpu);

  CpuState* cs = cpu_state(target);
  CafTask* prev = cs->current;
  if (prev) {
    CHECK(prev->oncpu());

    // We ping the agent on `target` below. Once that agent wakes up, it
    // automatically preempts `prev`. The kernel generates a TASK_PREEMPT
    // message for `prev`, which allows the scheduler to update the state for
    // `prev`.
    //
    // This also allows the scheduler to gracefully handle the case where `prev`
    // actually blocks/yields/etc. before it is preempted by the agent on
    // `target`. In any of those cases, a TASK_BLOCKED/TASK_YIELD/etc. message
    // is delivered for `prev` instead of a TASK_PREEMPT, so the state is still
    // updated correctly for `prev` even if it is not preempted by the agent.
  }

  SetGlobalCPU(target);
  enclave()->GetAgent(target)->Ping();

  // DLOG(INFO) << "Global agent moved from " << global_cpu.id()
  //            << " to CPU: " << target.id();
  return true;
}

std::unique_ptr<CafScheduler> SingleThreadCafScheduler(
    Enclave* enclave, CpuList cpulist, int32_t global_cpu,
    absl::Duration preemption_time_slice,
    absl::Duration reallocation_interval) {
  auto allocator = std::make_shared<SingleThreadMallocTaskAllocator<CafTask>>();
  auto scheduler = std::make_unique<CafScheduler>(
      enclave, std::move(cpulist), std::move(allocator), global_cpu,
      preemption_time_slice, reallocation_interval);
  return scheduler;
}

void CafAgent::AgentThread() {
  Channel& global_channel = global_scheduler_->GetDefaultChannel();
  gtid().assign_name("Agent:" + std::to_string(cpu().id()));
  if (verbose() > 1) {
    printf("Agent tid:=%d\n", gtid().tid());
  }
  SignalReady();
  WaitForEnclaveReady();

  PeriodicEdge debug_out(absl::Seconds(1));

  while (!Finished() || !global_scheduler_->Empty()) {
    BarrierToken agent_barrier = status_word().barrier();
    // Check if we're assigned as the Global agent.
    if (cpu().id() != global_scheduler_->GetGlobalCPUId()) {
      RunRequest* req = enclave()->GetRunRequest(cpu());

      if (verbose() > 1) {
        printf("Agent on cpu: %d Idled.\n", cpu().id());
      }
      req->LocalYield(agent_barrier, /*flags=*/0);
    } else {
      if (boosted_priority() &&
          global_scheduler_->PickNextGlobalCPU(agent_barrier, cpu())) {
        continue;
      }

      Message msg;
      while (!(msg = global_channel.Peek()).empty()) {
        global_scheduler_->DispatchMessage(msg);
        global_channel.Consume(msg);
      }

      global_scheduler_->ReallocateCores(false);

      global_scheduler_->GlobalSchedule(status_word(), agent_barrier);

      if (verbose() && debug_out.Edge()) {
        static const int flags =
            verbose() > 1 ? Scheduler::kDumpStateEmptyRQ : 0;
        if (global_scheduler_->debug_runqueue_) {
          global_scheduler_->debug_runqueue_ = false;
          global_scheduler_->DumpState(cpu(), Scheduler::kDumpAllTasks);
        } else {
          global_scheduler_->DumpState(cpu(), flags);
        }
      }
    }
  }
}

}  //  namespace ghost
