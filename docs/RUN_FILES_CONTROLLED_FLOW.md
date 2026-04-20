# Host-side controller: `run_files_controlled.py`

This document describes **`VM_sampler/VM_Capture_QEMU/run_files_controlled.py`** as the **host-side controller** of the active controlled-QEMU experiment flow. It is the orchestration entrypoint: it decides when the guest runs, when capture wraps a step, when the queue must drain, when offline analysis runs, and how per-step artifacts are separated on disk.

Evidence in this file is grounded in the implementation; anything that depends on other scripts or the guest environment is labeled **inferred** or **surrounding scripts**.

---

## Role in the architecture

| Aspect | Role |
|--------|------|
| **Position** | Top of the active pipeline for this project phase. Downstream components (QEMU capture launcher, producer, consumer, Rust delta, offline metrics script) are only engaged when this file enables them. |
| **Experiment control** | Enforces a **strict sequential** experiment: one **discrete step** at a time (one SSH command), with optional capture and post-step processing bound to that step. |
| **Boundary owner** | Owns **host** concerns: libvirt VM state, SSH to the guest, starting/stopping the capture processes around a step, polling the capture queue, invoking offline metrics, and rotating delta text files under `outputDir`. |
| **Non-responsibilities** | Does not implement memory dumps, delta math, or queue job processing; those live in the shell launcher, producer, consumer, and external binaries. |

---

## Sequence the controller enforces

The module docstring states the **base** loop:

```5:10:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
Flow per step:
  1) Ensure VM is running
  2) Wait for SSH reachability
  3) Run one command on VM over SSH and wait until it finishes
  4) Stop VM
  5) Move to next command
```

With **`CAPTURE_MODE`** (and optionally **`OFFLINE_METRICS_MODE`**), `main()` inserts capture and offline steps **after** the guest command and **before** `stop_vm()`. The enforced order per iteration is:

1. Derive `test_name` (e.g. `test1_run_idle` from `step_name_from_command`).
2. **`ensure_vm_running()`** — start or resume the libvirt domain.
3. **`wait_for_ssh()`** — poll until SSH accepts a trivial command.
4. If **`CAPTURE_MODE`**: build `step_matrix` path (`queueDir/run_matrix_<test_name>.npy`), call **`start_capture()`** (runs `run_qemu_capture.sh`), optional warmup sleep.
5. **`run(remote_cmd)`** — single quoted guest command over SSH; wait for exit.
6. If **`CAPTURE_MODE`**: **`stop_producer()`** → **`wait_for_queue_drain()`** → **`stop_consumer()`** → optional **`run_offline_step_metrics()`** → **`rotate_delta_files(test_name)`**.
7. **`stop_vm()`** — shutdown or destroy after timeout.
8. If the guest step exit code was non-zero, **abort** the whole run (after VM stop in the capture path).

```492:543:VM_sampler/VM_Capture_QEMU/run_files_controlled.py
        if CAPTURE_MODE:
            step_matrix = step_run_matrix_path(test_name)
            cap_rc, _ = start_capture(run_matrix_path=step_matrix)
            ...
        rc = run(f"{base} {shlex.quote(remote_cmd)}")
        ...
        if CAPTURE_MODE:
            stop_producer()
            drained = wait_for_queue_drain(...)
            ...
            stop_consumer()
            if OFFLINE_METRICS_MODE and step_matrix:
                run_offline_step_metrics(...)
            rotate_delta_files(test_name)
        stop_vm()
```

---

## How it starts the VM

`ensure_vm_running()` uses **`virsh domstate`** (via `virsh_state()`):

- If state contains **`running`** → return.
- If **`paused`** → **`virsh resume`**.
- Otherwise → **`virsh start`** for `VM_DOMAIN` on `VIRSH_URI`.

`virsh_state()` redirects `domstate` to **`/tmp/vm_state.txt`** and reads it back in Python.

---

## How it waits for readiness

Readiness is **SSH reachability**, not guest “fully booted” beyond that:

- **`wait_for_ssh()`** runs `ssh ... 'echo ok'` (or `sshpass` + `ssh` if `SSH_PASS` is set) in a loop until exit code 0 or **`SSH_WAIT_TIMEOUT`** expires.

There is **no** separate wait for services inside the guest except what `echo ok` implies.

---

## How it launches guest commands

- Builds **`ssh_base()`**: optional `sshpass`, `ssh` with `StrictHostKeyChecking=no`, optional `-i SSH_KEY`, optional `SSH_OPTS`, then **`SSH_TARGET`**.
- For each step, **`run(f"{base} {shlex.quote(remote_cmd)}")`** — exactly **one** remote command string per step.
- Commands come from **`load_steps()`**: either **`STEPS_FILE`** (line-based, `#` comments skipped) or the **embedded default** list pointing at `~/memorySignal/VM_executables/...`.

The host does **not** copy or validate guest paths; it only sends strings.

---

## How it coordinates capture start/stop

| Phase | Behavior |
|-------|----------|
| **Start** | **`start_capture()`** runs a shell: `cd CAPTURE_ROOT` then `CONFIG=… PRODUCER_SCRIPT=… BACKGROUND=1 ./run_qemu_capture.sh`, optionally `RUN_MATRIX=…`, `OFFLINE_MODE=1` when `OFFLINE_METRICS_MODE`, optional `BORG*` env. |
| **After guest step** | **`stop_producer()`** — `pkill -f capture_producer_qemu_pmemsave.sh` so no new dumps are queued for this step. |
| **Drain** | **`wait_for_queue_drain()`** — counts `*.json` under `queueDir/pending` and `queueDir/processing` until both are zero (or timeout). |
| **Stop consumer** | **`stop_consumer()`** — `pkill -f capture_consumer_qemu.sh` after drain. |

**Explicit in code:** only **`PRODUCER_SCRIPT`** is passed to the launcher; **`CONSUMER_SCRIPT`** is **not** set by Python — the consumer used is the **default** in `run_qemu_capture.sh` (**inferred** from that script).

---

## How it triggers offline metrics

Offline runs only when:

- **`OFFLINE_METRICS_MODE`** is true,
- **`CAPTURE_MODE`** is on (so `step_matrix` is set),
- **`run_offline_step_metrics()`** finds the script file, matrix file, and a resolved project root (`OFFLINE_PROJECT_ROOT` or `streaming.projectRoot` from the capture config).

It shells out to **`python3 OFFLINE_METRICS_SCRIPT`** with `--matrix`, `--step-name`, `--output-root`, `--project-root`, `--baseline-dir`, window/step sizes, and **`--is-baseline`** when `i == BASELINE_STEP_NUMBER`.

**Surrounding behavior:** what `offline_step_metrics.py` writes under `offline/<step_name>/` is **explicit** in that script; this controller only passes paths and flags.

---

## How it organizes outputs per test step

| Mechanism | Purpose |
|-----------|---------|
| **`test_name`** | `f"test{i}_{step_label}"` — unique label per step (e.g. `test1_run_idle`). |
| **`RUN_MATRIX`** | Points the consumer at **`queueDir/run_matrix_<test_name>.npy`** so each step’s matrix is isolated when capture is on. |
| **`rotate_delta_files(test_name)`** | Moves `outputDir/hamming/*.txt` and `outputDir/cosine/*.txt` into **`outputDir/rotated/<test_name>/hamming|cosine/`** (rename; collision adds a timestamp suffix). |

Offline outputs use **`--step-name`** (same `test_name`) and default baseline dir under **`output_root/offline/baseline`** unless overridden.

---

## Assumptions about the environment

**Explicit in code (behavioral assumptions):**

- **`SSH_TARGET`** is set and reachable from the host.
- **`virsh`** can control **`VM_DOMAIN`** on **`VIRSH_URI`**.
- **`libvirt`**, **`ssh`** (and optionally **`sshpass`**) exist on the host.
- When capture is on: **`CAPTURE_CONFIG`** is readable JSON with **`queueDir`** and **`outputDir`**; **`CAPTURE_ROOT`** contains **`run_qemu_capture.sh`** and the chosen producer script.
- **`run_qemu_capture.sh`** with `BACKGROUND=1` creates **`capture_pids.txt`** in `CAPTURE_ROOT` (optional read).

**Inferred (not verified by this file):**

- Guest has workload scripts at the paths embedded in the default commands (e.g. **`~/memorySignal/VM_executables/...`**).
- Consumer is **`capture_consumer_qemu.sh`** via launcher default.
- Downstream tools (Rust delta, Python imports inside offline metrics) succeed at the paths in config.

---

## External script dependencies

| Dependency | How it is invoked |
|------------|-------------------|
| **`run_qemu_capture.sh`** | `os.system` from `start_capture()` under `CAPTURE_ROOT` |
| **`CAPTURE_PRODUCER_SCRIPT`** (default `capture_producer_qemu_pmemsave.sh`) | Passed as env to launcher; launcher starts it |
| **`capture_consumer_qemu.sh`** | **Not** passed by Python; **default** in `run_qemu_capture.sh` |
| **`offline_step_metrics.py`** (or `OFFLINE_METRICS_SCRIPT`) | `python3 …` from `run_offline_step_metrics()` |
| **`virsh`** | `domstate`, `resume`, `start`, `shutdown`, `destroy` |
| **`ssh` / `sshpass`** | Guest command execution |
| **`pkill`** | Stop producer/consumer by process name pattern |

---

## Input configuration points

| Source | Variables / files |
|--------|-------------------|
| **Environment** | `VIRSH_URI`, `VM_DOMAIN`, `SSH_*`, `STEPS_FILE`, `TEST_EXEC_SECONDS`, `CAPTURE_MODE`, `CAPTURE_ROOT`, `CAPTURE_CONFIG`, `CAPTURE_PRODUCER_SCRIPT`, `CAPTURE_WARMUP_SECONDS`, `OFFLINE_*`, `BASELINE_STEP_NUMBER`, `QUEUE_DRAIN_TIMEOUT_MINUTES`, `BORG*`, `FORCE_DESTROY`, `STOP_TIMEOUT` |
| **JSON file** | `CAPTURE_CONFIG` — read for `queueDir`, `outputDir`, `streaming.projectRoot` |
| **Optional** | `STEPS_FILE` — host path to step lines |

---

## Outputs it expects and moves

| Output | Producer |
|--------|----------|
| **`/tmp/vm_state.txt`** | Written by `virsh` redirect; read by Python |
| **`capture_pids.txt`** | Written by **`run_qemu_capture.sh`** when background; optionally read |
| **Queue drain** | Expects consumer to empty `pending`/`processing` JSON counts |
| **Step matrix** | Expects path `queueDir/run_matrix_<test_name>.npy` to exist for offline when enabled |
| **Delta text files** | Expects `outputDir/hamming/*.txt` and `outputDir/cosine/*.txt` before rotation; **moves** them to `outputDir/rotated/<test_name>/...` |
| **`outputDir` dirs** | **Creates** `outputDir` and rotated dirs as needed via `rotate_delta_files` |

The controller does **not** write queue dumps or matrix contents; it only sets **`RUN_MATRIX`** and rotates text files.

---

## Explicit in code vs inferred from surrounding scripts

| Topic | Explicit in `run_files_controlled.py` | Inferred from elsewhere |
|-------|--------------------------------------|-------------------------|
| Step order, SSH, virsh, capture env, pkill, offline CLI | Yes | — |
| `consumer` process name in `pkill` | Yes (`capture_consumer_qemu.sh`) | Matches launcher default |
| Which consumer script starts | No | `run_qemu_capture.sh` defaults `CONSUMER_SCRIPT` |
| Producer log / consumer log paths | No | `run_qemu_capture.sh` writes `producer.log` / `consumer.log` under `CAPTURE_ROOT` |
| Dump paths, job JSON, Rust binary | No | `config_qemu_upc.json` + producer/consumer |
| Guest file layout | No | Default SSH strings only |

---

## Dead code (not part of the active loop)

`pause_capture_processes()`, `resume_capture_processes()`, `stop_capture()` are defined but **`main()`** does not call them. Treat them as non-architecture unless a future change wires them in.

---

## See also

- [`ACTIVE_PIPELINE_FILE_MAP.md`](ACTIVE_PIPELINE_FILE_MAP.md) — exact file read/write and directory map from this entrypoint
- [`QEMU_CAPTURE_PIPELINE.md`](QEMU_CAPTURE_PIPELINE.md) — producer and consumer behavior after the launcher starts
- [`OFFLINE_METRICS_AND_OUTPUTS.md`](OFFLINE_METRICS_AND_OUTPUTS.md) — offline outputs and directory conventions
