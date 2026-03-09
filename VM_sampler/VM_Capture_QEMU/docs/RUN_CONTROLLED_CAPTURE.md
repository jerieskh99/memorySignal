# Controlled QEMU Capture Run Guide

This guide explains how to run the QEMU workflow with:

- VM workload control (`run_files_controlled.py`)
- optional capture per workload phase (`CAPTURE_MODE=1`)
- producer/consumer pipeline (`run_qemu_capture.sh`)

---

## 1) What each script does

- `run_files_controlled.py` (host-side orchestrator)
  - Starts/resumes VM
  - Waits for SSH
  - Runs one workload command on VM
  - Optionally starts/stops capture around each workload (`CAPTURE_MODE=1`)
  - Stops VM
  - Repeats for all steps

- `run_qemu_capture.sh` (capture launcher)
  - Starts producer + consumer
  - Foreground or background mode
  - Supports optional one-shot SSH command mode

- `capture_producer_qemu_pmemsave.sh`
  - Suspends VM, runs `pmemsave`, resumes VM, enqueues jobs

- `capture_consumer_qemu.sh`
  - Reads jobs, runs Rust delta, appends run matrix, optional streaming metrics
  - Deletes only `prev` in rolling chain mode

---

## 2) Prerequisites

On the host:

- `virsh` access to the VM domain (default: `qemu:///system`)
- `ssh` client
- Python 3
- `run_qemu_capture.sh` and related scripts executable

On the VM:

- Workload scripts available under `~/VM_executables` (or your chosen path)
- SSH reachable from host (`SSH_TARGET=user@vm-ip`)

Capture prerequisites:

- `config_qemu_upc.json` (or your config file) points to valid paths
- Rust delta binary exists at `rustDeltaCalculationProgram`

---

## 3) Run modes

### A) Workload control only (no capture)

```bash
cd ~/memorySignal/VM_sampler/VM_Capture_QEMU
SSH_TARGET=user@vm-ip ./run_files_controlled.py
```

### B) Workload control + capture mode

```bash
cd ~/memorySignal/VM_sampler/VM_Capture_QEMU
SSH_TARGET=user@vm-ip CAPTURE_MODE=1 ./run_files_controlled.py
```

With capture mode, each step does:

1. Start VM + wait SSH
2. Start capture in background
3. Run VM command
4. Stop capture
5. Stop VM

---

## 4) `run_files_controlled.py` environment variables

### Required

- `SSH_TARGET`
  - Example: `jeries@192.168.122.50`

### VM control

- `VM_DOMAIN` (default: `Kali Jeries`)
- `VIRSH_URI` (default: `qemu:///system`)
- `STOP_TIMEOUT` (default: `60`)
- `FORCE_DESTROY` (default: `1`)
  - `1/true/yes`: force `virsh destroy` if graceful shutdown times out

### SSH behavior

- `SSH_KEY` (optional private key path)
- `SSH_OPTS` (optional extra options, e.g. `-p 2222`)
- `SSH_WAIT_TIMEOUT` (default: `120`)

### Step source

- `STEPS_FILE` (optional)
  - If provided, each non-empty, non-comment line is one remote command.
  - If not provided, built-in default step sequence is used.

### Capture integration (optional)

- `CAPTURE_MODE` (default: `0`)
  - `1/true/yes` enables capture around each step
- `CAPTURE_ROOT`
  - Default: directory containing `run_files_controlled.py`
- `CAPTURE_CONFIG`
  - Default: `${CAPTURE_ROOT}/config_qemu_upc.json`
- `CAPTURE_PRODUCER_SCRIPT`
  - Default: `${CAPTURE_ROOT}/capture_producer_qemu_pmemsave.sh`
- `CAPTURE_WARMUP_SECONDS` (default: `2`)

Example:

```bash
SSH_TARGET=user@vm-ip \
CAPTURE_MODE=1 \
CAPTURE_CONFIG=./config_qemu_upc.json \
CAPTURE_WARMUP_SECONDS=3 \
./run_files_controlled.py
```

---

## 5) `run_qemu_capture.sh` environment variables

- `CONFIG` (default: `./config_qemu.json`)
- `PRODUCER_SCRIPT` (default: pmemsave producer)
- `CONSUMER_SCRIPT` (default: qemu consumer)
- `BACKGROUND` (`1` to run both in background)

SSH utility mode in this script (optional):

- `SSH_TARGET`, `SSH_COMMAND`, `SSH_KEY`, `SSH_OPTS`
- `SSH_ONLY=1` -> run SSH command and exit
- `SSH_BEFORE_START=1` -> run SSH command then start capture

Example capture start:

```bash
CONFIG=./config_qemu_upc.json \
PRODUCER_SCRIPT=$PWD/capture_producer_qemu_pmemsave.sh \
BACKGROUND=1 \
./run_qemu_capture.sh
```

---

## 6) Custom step file example (`STEPS_FILE`)

Create `steps_controlled.txt`:

```text
# idle
bash ~/VM_executables/run_idle.sh --time 30

# memory workloads
python3 ~/VM_executables/mem_stream.py --mb 128 --seconds 120
python3 ~/VM_executables/mem_pointer_chase.py --mb 1024 --seconds 120 --seed 123

# io workloads
python3 ~/VM_executables/io_seq_fsync.py --seconds 120 --kb 4096 --fsync-wait 1 --path io_seq.bin
```

Run:

```bash
SSH_TARGET=user@vm-ip STEPS_FILE=./steps_controlled.txt CAPTURE_MODE=1 ./run_files_controlled.py
```

---

## 7) Troubleshooting

- `env: 'bash\r': No such file or directory`
  - Convert scripts to LF on server (CRLF issue), or run `fix_line_endings_server.sh`.

- `No such file or directory` for Rust binary
  - Fix `rustDeltaCalculationProgram` path in config, and/or build Rust binary.

- Pending queue grows
  - Producer faster than consumer (especially after streaming starts).
  - Increase `intervalMsec`, lower backpressure threshold, or disable streaming for debugging.

- Permission denied deleting dumps
  - Use sudo-enabled deletion path in consumer and warm sudo timestamp (`sudo -v`) before runs.

