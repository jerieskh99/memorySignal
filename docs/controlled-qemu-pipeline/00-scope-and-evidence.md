# Active Controlled QEMU Pipeline

## Purpose
This booklet documents only the currently active controlled QEMU pipeline used in the present project phase. It starts from the host-side controller in `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` and follows only the files that this active flow directly invokes or clearly requires.

The aim is architectural accuracy, not repository-wide coverage. When the code, config, and nearby notes disagree, this booklet favors the active implementation and labels the mismatch explicitly.

## Scope Boundary
### In Scope
- Host orchestration in `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- Capture launcher in `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- Active producer in `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
- Active consumer in `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- Active capture configuration in `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- Guest workloads directly referenced by the default controller step list in `VM_executables/`
- Conditional offline post-step metrics in `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`

### Out Of Scope
- Alternate producers not selected by the active controller defaults: `VM_sampler/VM_Capture_QEMU/capture_producer_qemu.sh` and `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_user_raw.sh`
- Helper and cleanup scripts not called by the active controller: `VM_sampler/VM_Capture_QEMU/cleanup_qemu_capture.sh` and `VM_sampler/VM_Capture_QEMU/fix_line_endings_server.sh`
- Legacy or alternative notes that describe other paths rather than the active flow, including `VM_sampler/VM_Capture_QEMU/README.md` and `VM_sampler/VM_Capture_QEMU/RAW_CAPTURE_ALTERNATIVE.md`
- `VM_executables/run_files.sh`, which is present but not used by the default `run_files_controlled.py` loop
- Internal implementations of external metric packages unless directly visible from the active scripts

## Evidence Policy
### Directly Supported
Claims are marked directly supported when they come from the active code or active config, for example:
- the per-step lifecycle in `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- the producer and consumer startup logic in `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- the queue, dump, and output directories defined in `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`

### Inferred
Claims are marked inferred when they are reasonable consequences of the active files but are not asserted by the controller itself. Example:
- the controller never passes `CONSUMER_SCRIPT`, but `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh` defaults it to `capture_consumer_qemu.sh`, so consumer participation is inferred from launcher defaults rather than explicitly selected in Python

### Ambiguous Or Unverified
Claims remain ambiguous when the repository shows an assumption but does not prove it in the active path. Examples:
- the guest is assumed to contain workloads under `~/memorySignal/VM_executables/...`, but the host controller does not validate that guest-side layout
- nearby documentation still mentions `~/VM_executables/...` in some examples, which conflicts with the active command strings in `run_files_controlled.py`

## Active Flow Summary
The active architecture is a host-controlled step runner wrapped around an optional producer-consumer memory capture pipeline:

1. The host ensures the VM is running and reachable over SSH.
2. The host runs exactly one guest workload command per step.
3. If capture is enabled, the host starts the QEMU capture launcher before the guest step.
4. The producer repeatedly acquires RAW memory dumps and writes queue jobs.
5. The consumer converts each dump pair into delta outputs, appends a per-frame vector into a run matrix, and may trigger live streaming metrics.
6. After the guest step finishes, the host stops the producer, waits for queue drain, stops the consumer, powers the VM down, optionally runs offline step metrics, and rotates delta outputs.

## Booklet Map
- [`../ACTIVE_PIPELINE_FILE_MAP.md`](../ACTIVE_PIPELINE_FILE_MAP.md): conservative file-and-folder map, dependency map, flow map, used vs excluded files (traced from `run_files_controlled.py` only)
- `01-host-orchestrator.md`: top-level host control flow and step lifecycle
- `02-capture-launch-and-producer.md`: shell handoff and RAW dump production
- `03-consumer-and-run-matrix.md`: queue processing, delta outputs, run-matrix accumulation, and streaming
- `04-guest-workloads.md`: default guest workloads and optional `STEPS_FILE` variants
- `05-offline-metrics.md`: conditional per-step offline metrics stage
- `06-inputs-outputs-and-directories.md`: path conventions and artifact movement
- `07-ambiguities-and-out-of-scope.md`: direct support, inference, unresolved gaps, and explicit exclusions
