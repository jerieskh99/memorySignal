# Host Orchestrator

> **Canonical reference:** the full host-controller section lives in [`../RUN_FILES_CONTROLLED_FLOW.md`](../RUN_FILES_CONTROLLED_FLOW.md). This chapter is a shorter companion; use the linked doc for architecture role, sequence, assumptions, explicit vs inferred, and I/O.

## Purpose
This section documents `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` as the host-side orchestrator for the active controlled QEMU workflow. It is the top-level entrypoint for the current phase and determines when the VM runs, when capture is started, which guest workloads execute, and when per-step outputs are finalized.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`

## Support Level
- Directly supported by the controller implementation

## Out Of Scope
- Producer internals beyond the callout to `run_qemu_capture.sh`
- Consumer internals beyond queue-drain coordination
- Mathematical internals of downstream metric modules

## Controller Role
The controller is a sequential step runner. Its module docstring states the intended lifecycle per step:

```5:10:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
Flow per step:
  1) Ensure VM is running
  2) Wait for SSH reachability
  3) Run one command on VM over SSH and wait until it finishes
  4) Stop VM
  5) Move to next command
```

In practice, the active implementation extends that loop with optional capture and optional offline metrics.

## VM And SSH Control
The controller uses `virsh` for host-side VM state control and `ssh` or `sshpass + ssh` for guest command execution.

- `ensure_vm_running()` checks `virsh domstate`, resumes a paused domain, or starts a stopped domain
- `wait_for_ssh()` polls `ssh ... 'echo ok'` until the guest becomes reachable
- `stop_vm()` issues `virsh shutdown`, waits up to `STOP_TIMEOUT`, and can fall back to `virsh destroy` when `FORCE_DESTROY` is enabled

These functions make the host controller, rather than the guest workload scripts, responsible for VM lifecycle boundaries.

## Step Source
The controller supports two step sources:

- Default active sequence embedded in `load_steps()`
- External line-based step file selected by `STEPS_FILE`

If `STEPS_FILE` is unset, the default active sequence is used:

```449:457:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
    return [
        "bash ~/memorySignal/VM_executables/run_idle.sh --time 30",
        f"python3 ~/memorySignal/VM_executables/mem_stream.py --mb 128 --seconds {TEST_EXEC_SECONDS}",
        f"python3 ~/memorySignal/VM_executables/mem_pointer_chase.py --mb 1024 --seconds {TEST_EXEC_SECONDS} --seed 123",
        f"python3 ~/memorySignal/VM_executables/mem_alloc_touch_pages.py --objects 2000 --object-kb 256 --sleep-ms 20 --seconds {TEST_EXEC_SECONDS}",
        f"python3 ~/memorySignal/VM_executables/io_seq_fsync.py --seconds {TEST_EXEC_SECONDS} --kb 4096 --fsync-wait 1 --path io_seq.bin",
        f"python3 ~/memorySignal/VM_executables/io_rand_rw.py --seconds {TEST_EXEC_SECONDS} --file-mb 2048 --block-kb 64 --write-ratio 0.5 --path io_rand.bin --seed 123",
        f"python3 ~/memorySignal/VM_executables/io_many_files.py --seconds {TEST_EXEC_SECONDS} --files-per-batch 500 --payload-bytes 1024 --seed 123",
    ]
```

## Per-Step Lifecycle
For each step, the controller derives a test label such as `test1_run_idle`, then executes the following active lifecycle:

1. Ensure the VM is running.
2. Wait for SSH reachability.
3. If `CAPTURE_MODE=1`, derive a step-specific `run_matrix_<test>.npy` path and start capture.
4. Run one remote command over SSH.
5. If capture is active, stop only the producer first.
6. Poll the capture queue until `pending` and `processing` are both empty.
7. Stop the consumer.
8. Stop the VM before host-side post-processing.
9. If `OFFLINE_METRICS_MODE=1`, invoke `offline_step_metrics.py` on the step-specific run matrix.
10. Rotate delta text outputs into `outputDir/rotated/<test_name>/...`.
11. Abort the sequence if the guest command failed.

This extended lifecycle is directly visible in the main loop:

```492:543:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
        if CAPTURE_MODE:
            step_matrix = step_run_matrix_path(test_name)
            cap_rc, _ = start_capture(run_matrix_path=step_matrix)
            if cap_rc != 0:
                print(f"[CONTROL] ERROR: failed to start capture (exit={cap_rc}).")
                return cap_rc
            if CAPTURE_WARMUP_SECONDS > 0:
                print(f"[CONTROL] Capture warmup: sleeping {CAPTURE_WARMUP_SECONDS}s")
                time.sleep(CAPTURE_WARMUP_SECONDS)

        print("[CONTROL] Running command over SSH...")
        rc = run(f"{base} {shlex.quote(remote_cmd)}")
        rc = rc >> 8

        if CAPTURE_MODE:
            stop_producer()
            drained = wait_for_queue_drain(timeout_minutes=QUEUE_DRAIN_TIMEOUT_MINUTES)
            if not drained:
                stop_consumer()
                stop_vm()
                return 1

            stop_consumer()

            stop_vm()

            if OFFLINE_METRICS_MODE and step_matrix:
                run_offline_step_metrics(
                    step_name=test_name,
                    matrix_path=step_matrix,
                    is_baseline=is_baseline_step,
                )

            rotate_delta_files(test_name)

        if not vm_stopped:
            stop_vm()
```

## Environment Variables That Change The Active Flow
The most important controller parameters are:

- `SSH_TARGET`: required guest SSH destination
- `VM_DOMAIN` and `VIRSH_URI`: host-side libvirt target
- `STEPS_FILE`: optional external step list
- `TEST_EXEC_SECONDS`: default duration inserted into the built-in Python workload commands
- `CAPTURE_MODE`: enables the producer-consumer capture path
- `CAPTURE_CONFIG`: points to the active capture config, defaulting to `config_qemu_upc.json`
- `CAPTURE_PRODUCER_SCRIPT`: defaults to the pmemsave producer
- `OFFLINE_METRICS_MODE`: enables per-step offline metrics after queue drain, consumer stop, and VM shutdown
- `BASELINE_STEP_NUMBER`: determines which step is treated as the clean baseline for offline PLV reuse

## Directly Implemented Versus Inferred
### Directly Implemented
- One guest command is executed per step over SSH
- The VM is started before each step and stopped after each step
- Capture start and stop are controlled from the host when `CAPTURE_MODE=1`
- Offline metrics are triggered only after queue drain, consumer stop, and VM shutdown when `OFFLINE_METRICS_MODE=1`

### Inferred Or Conditional
- The guest-side path `~/memorySignal/VM_executables/...` is assumed by the command strings, but not validated by the controller
- The specific consumer script used during capture is not selected in Python; it is inherited from the launcher defaults described in the next section

## Exclusions For This Section
This section does not treat unused helpers such as `pause_capture_processes()` and `resume_capture_processes()` as active architecture. They exist in the file but are not called by the current main loop.
