#include "schedulers/shenanigan/shenanigan_scheduler.h"

namespace ghost {

ShenaniganScheduler::ShenaniganScheduler(
    Enclave* enclave, CpuList cpulist,
    std::shared_ptr<TaskAllocator<ShenaniganTask>> allocator,
    int32_t global_cpu, absl::Duration reallocation_interval)
    : BasicDispatchScheduler(enclave, std::move(cpulist), std::move(allocator)),
      global_cpu_(global_cpu),
      global_channel_(GHOST_MAX_QUEUE_ELEMS, 0),
      reallocation_interval_(reallocation_interval) {
  if (!cpus().IsSet(global_cpu)) {
    Cpu c = cpus().Front();
    CHECK(c.valid());
    global_cpu_ = c.id();
  }
}

ShenaniganScheduler::~ShenaniganScheduler() {}

void ShenaniganScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    cs->agent = enclave()->GetAgent(cpu);
    CHECK_NE(cs->agent, nullptr);
  }
}

bool ShenaniganScheduler::Available(const Cpu& cpu) {
  CpuState* cs = cpu_state(cpu);

  if (cs->agent) return cs->agent->cpu_avail();

  return false;
}

void ShenaniganScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state       rq_pos  P\n");
  allocator()->ForEachTask([](Gtid gtid, const ShenaniganTask* task) {
    absl::FPrintF(stderr, "%-12s%-12s%d\n", gtid.describe(),
                  ShenaniganTask::RunStateToString(task->run_state),
                  task->cpu.valid() ? task->cpu.id() : -1);
    return true;
  });
}

void ShenaniganScheduler::DumpState(const Cpu& agent_cpu, int flags) {
  if (flags & kDumpAllTasks) {
    DumpAllTasks();
  }

  if (!(flags & kDumpStateEmptyRQ)) {  //&&  RunqueueEmpty()) {
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
  // fprintf(stderr, " rq_l=%ld", RunqueueSize());
  fprintf(stderr, "\n");
}

ShenaniganScheduler::CpuState* ShenaniganScheduler::cpu_state_of(const ShenaniganTask* task) {
  CHECK(task->cpu.valid());
  CHECK(task->oncpu());
  CpuState* cs = cpu_state(task->cpu);
  CHECK(task == cs->current);
  return cs;
}

// Event Handler Functions

void ShenaniganScheduler::TaskNew(ShenaniganTask* task, const Message& msg) {
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());

  task->seqnum = msg.seqnum();
  task->run_state = ShenaniganTask::RunState::kBlocked;

  const Gtid gtid(payload->gtid);
  task->vm_id = gtid.tgid();

  if (payload->runnable) {
    task->run_state = ShenaniganTask::RunState::kRunnable;
    Enqueue(task);
  }

  num_tasks_++;
}

void ShenaniganScheduler::TaskRunnable(ShenaniganTask* task,
                                       const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());

  task->run_state = ShenaniganTask::RunState::kRunnable;
  task->prio_boost = !payload->deferrable;
  Enqueue(task);
}

void ShenaniganScheduler::TaskDeparted(ShenaniganTask* task,
                                       const Message& msg) {
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

void ShenaniganScheduler::TaskDead(ShenaniganTask* task, const Message& msg) {
  CHECK_EQ(task->run_state, ShenaniganTask::RunState::kBlocked);
  allocator()->FreeTask(task);
  num_tasks_--;
}

void ShenaniganScheduler::TaskBlocked(ShenaniganTask* task,
                                      const Message& msg) {
  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(task->queued());
    RemoveFromRunqueue(task);
  }
  task->run_state = ShenaniganTask::RunState::kBlocked;
}

void ShenaniganScheduler::TaskPreempted(ShenaniganTask* task,
                                        const Message& msg) {
  task->preempted = true;
  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
    task->run_state = ShenaniganTask::RunState::kRunnable;
    Enqueue(task);
  } else {
    CHECK(task->queued());
  }
}

void ShenaniganScheduler::TaskYield(ShenaniganTask* task, const Message& msg) {
  if (task->oncpu()) {
    CpuState* cs = cpu_state_of(task);
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
    Yield(task);
  } else {
    CHECK(task->queued());
  }
}

void ShenaniganScheduler::Yield(ShenaniganTask* task) {
  CHECK(task->oncpu() || task->runnable());
  task->run_state = ShenaniganTask::RunState::kYielding;
  yielding_tasks_.push_back(task);
}

void ShenaniganScheduler::Unyield(ShenaniganTask* task) {
  CHECK(task->yielding());

  auto it = std::find(yielding_tasks_.begin(), yielding_tasks_.end(), task);
  CHECK(it != yielding_tasks_.end());
  yielding_tasks_.erase(it);

  Enqueue(task);
}

void ShenaniganScheduler::Enqueue(ShenaniganTask* task) {
  CHECK_EQ(task->run_state, ShenaniganTask::RunState::kRunnable);
  task->run_state = ShenaniganTask::RunState::kQueued;

  auto vm_id = task->vm_id;
  auto it = run_queues_.find(vm_id);

  if (it == run_queues_.end()) {
    GHOST_DPRINT(1, stderr, "Created run queue for VM %d", vm_id);
    run_queues_.emplace(vm_id, std::deque<ShenaniganTask*>());
  } else {
    GHOST_DPRINT(1, stderr, "Enqueued task %s for VM %d", task->gtid.describe(),
                 vm_id);
    std::deque<ShenaniganTask*>& run_queue = it->second;
    if (task->prio_boost || task->preempted) {
      run_queue.push_front(task);
    } else {
      run_queue.push_back(task);
    }
  }
}

ShenaniganTask* ShenaniganScheduler::Dequeue(pid_t vm_id) {
  // no VMs to schedule
  if (run_queues_.size() == 0) {
    return nullptr;
  }

  auto run_queue = run_queues_[vm_id];
  ShenaniganTask* task = run_queue.front();
  CHECK_EQ(task->run_state, ShenaniganTask::RunState::kQueued);
  task->run_state = ShenaniganTask::RunState::kRunnable;
  run_queue.pop_front();

  return task;
}

void ShenaniganScheduler::RemoveFromRunqueue(ShenaniganTask* task) {
  CHECK(task->queued());

  auto run_queue_ = run_queues_[task->vm_id];

  for (int pos = run_queue_.size() - 1; pos >= 0; pos--) {
    // The [] operator for 'std::deque' is constant time
    if (run_queue_[pos] == task) {
      // Caller is responsible for updating 'run_state' if task is
      // no longer runnable.
      task->run_state = ShenaniganTask::RunState::kRunnable;
      run_queue_.erase(run_queue_.cbegin() + pos);
      return;
    }
  }

  // This state is unreachable because the task is queued.
  CHECK(false);
}

void ShenaniganScheduler::TaskIsOnCpu(ShenaniganTask* task, const Cpu& cpu) {
  CpuState* cs = cpu_state(cpu);
  CHECK_EQ(task, cs->current);

  GHOST_DPRINT(3, stderr, "Task %s is on cpu %d", task->gtid.describe(),
               cpu.id());

  task->run_state = ShenaniganTask::RunState::kOnCpu;
  task->cpu = cpu;
  task->preempted = false;
  task->prio_boost = false;
}

void ShenaniganScheduler::GlobalSchedule(const StatusWord& agent_sw,
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

    if (cs->current) {
      // This CPU is running a task, so don't schedule anything on it.
      continue;
    }
    // No task is running on this CPU, so it's available for scheduling.
    available.Set(cpu);
  }

  for (auto& [vm_id, run_queue] : run_queues_) {
    // auto cores = cores_per_v
    auto [iter, created] =
        cores_per_vm_.try_emplace(vm_id, topology()->EmptyCpuList());

    if (created) {
      GHOST_DPRINT(1, stderr, "Created empty core list for VM %d", vm_id);
    }

    CpuList& cores = iter->second;

    if (run_queue.empty()) {
      GHOST_DPRINT(1, stderr, "No tasks for VM %d, parking all cores assigned.",
                   vm_id);
      cores = topology()->EmptyCpuList();
    } else {
      // runnable vCPUs
      if (cores.Empty()) {
        // assign single core to this VM
        // TODO: (vjabrayilov) This is a temporary solution.
        // We need to implement algorithm to assign cores considering vCPU count
        // and current runnable vCPUs
        const Cpu& next_cpu = available.Front();
        cores.Set(next_cpu);
        available.Clear(next_cpu);
        GHOST_DPRINT(1, stderr, "Assigned core %d to VM %d", next_cpu.id(),
                     vm_id);
      }

      for (const Cpu& next_cpu : cores) {
        // Dequeue the next task from runqeue to run on this CPU.
        ShenaniganTask* next = Dequeue(vm_id);

        if (!next) {
          // No more queued vCPUs for this VM
          // TODO: (vjabrayilov) Rest of the cores may or may not be parked;
          // discuss tradeoffs.
          break;
        }

        // If `next->status_word.on_cpu()` is true, then `next` was previously
        // preempted by this scheduler but hasn't been moved off the CPU it was
        // previously running on yet.
        //
        // If `next->seqnum != next->status_word.barrier()` is true, then there
        // are pending messages for `next` that we have not read yet. Thus, do
        // not schedule `next` since we need to read the messages. We will
        // schedule `next` in a future iteration of the global scheduling loop.
        // TODO: (vjabrayilov) Copy-pasted from  ghOSt FIFO, not sure if it's
        // needed.
        if (next->status_word.on_cpu() ||
            next->seqnum != next->status_word.barrier()) {
          Yield(next);
          continue;
        }

        CpuState* cs = cpu_state(next_cpu);
        CHECK_EQ(cs->current, nullptr);
        cs->current = next;

        available.Clear(next_cpu);
        assigned.Set(next_cpu);

        RunRequest* req = enclave()->GetRunRequest(next_cpu);
        req->Open(
            {.target = next->gtid,
             .target_barrier = next->seqnum,
             // No need to set `agent_barrier` because the agent barrier is
             // not checked when a global agent is scheduling a CPU other than
             // the one that the global agent is currently running on.
             .commit_flags = COMMIT_AT_TXN_COMMIT});
      }
    }
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
      // The transaction succeeded, so the task is now running on `next_cpu`.
      TaskIsOnCpu(cs->current, next_cpu);
    } else {
      GHOST_DPRINT(3, stderr, "Failed to commit (state=%d)", req->state());

      cs->current->prio_boost = true;
      Enqueue(cs->current);
      cs->current = nullptr;
    }
  }

  // Yielding tasks are moved back to the runqueue having skipped one round
  // of scheduling decisions.
  if (!yielding_tasks_.empty()) {
    for (ShenaniganTask* t : yielding_tasks_) {
      CHECK_EQ(t->run_state, ShenaniganTask::RunState::kYielding);
      t->run_state = ShenaniganTask::RunState::kRunnable;
      Enqueue(t);
    }
    yielding_tasks_.clear();
  }
}

bool ShenaniganScheduler::PickNextGlobalCPU(BarrierToken agent_barrier,
                                            const Cpu& this_cpu) {
  Cpu target(Cpu::UninitializedType::kUninitialized);
  Cpu global_cpu = topology()->cpu(GetGlobalCPUId());
  int numa_node = global_cpu.numa_node();

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
  ShenaniganTask* prev = cs->current;
  if (prev) {
    CHECK(prev->oncpu());

    // We ping the agent on `target` below. Once that agent wakes up, it
    // automatically preempts `prev`. The kernel generates a TASK_PREEMPT
    // message for `prev`, which allows the scheduler to update the state for
    // `prev`.
    //
    // This also allows the scheduler to gracefully handle the case where
    // `prev` actually blocks/yields/etc. before it is preempted by the agent
    // on `target`. In any of those cases, a TASK_BLOCKED/TASK_YIELD/etc.
    // message is delivered for `prev` instead of a TASK_PREEMPT, so the state
    // is still updated correctly for `prev` even if it is not preempted by
    // the agent.
  }

  SetGlobalCPU(target);
  enclave()->GetAgent(target)->Ping();

  return true;
}

std::unique_ptr<ShenaniganScheduler> SingleThreadShenaniganScheduler(
    Enclave* enclave, CpuList cpulist, int32_t global_cpu,
    absl::Duration reallocation_interval) {
  auto allocator =
      std::make_shared<SingleThreadMallocTaskAllocator<ShenaniganTask>>();
  auto scheduler = std::make_unique<ShenaniganScheduler>(
      enclave, std::move(cpulist), std::move(allocator), global_cpu,
      reallocation_interval);
  return scheduler;
}

void ShenaniganAgent::AgentThread() {
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
}  // namespace ghost