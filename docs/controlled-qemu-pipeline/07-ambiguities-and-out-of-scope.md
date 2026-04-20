# Ambiguities And Out Of Scope

## Purpose
This section records the boundary cases, inferred assumptions, current mismatches, and explicit exclusions needed to keep the booklet scientifically accurate.

## Relevant Files
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- `VM_sampler/VM_Capture_QEMU/docs/RUN_CONTROLLED_CAPTURE.md`
- `VM_sampler/VM_Capture_QEMU/README.md`

## Support Level
- Mixed by design: this chapter classifies evidence rather than asserting a single behavior source

## Out Of Scope
- Resolving unsupported claims by speculation
- Expanding the documented architecture to other `VM_sampler/` branches

## Directly Supported Active Facts
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py` is the active orchestration entrypoint for the current documentation phase
- The controller executes one guest command per step over SSH
- The active default capture path uses `capture_producer_qemu_pmemsave.sh`
- The launcher starts both producer and consumer in background mode when invoked by the controller
- The consumer builds a delta-derived run matrix and can trigger streaming metrics
- `offline_step_metrics.py` is called only after queue drain when offline mode is enabled
- `config_qemu_upc.json` is the active default config path for the controller

## Inferred But Reasonable Claims
- The active consumer in the controller path is `capture_consumer_qemu.sh`, inferred from the launcher's default `CONSUMER_SCRIPT`
- The guest contains a checkout or copy of the repository under `~/memorySignal/`, inferred from the controller's command strings
- The Rust delta program and the external metric modules are available and callable at the configured paths, inferred from configuration and call sites rather than validated in the controller

## Ambiguities And Mismatches
### Guest Workload Path Mismatch
Active code:
- `run_files_controlled.py` and the optional `steps_*.txt` files use `~/memorySignal/VM_executables/...`

Nearby documentation:
- `VM_sampler/VM_Capture_QEMU/docs/RUN_CONTROLLED_CAPTURE.md` still shows `~/VM_executables/...` in some examples

Implication:
- the booklet should treat the controller command strings as authoritative for the current active flow and note the documentation mismatch explicitly

### Consumer Selection Is Indirect
Active code:
- the controller passes only `PRODUCER_SCRIPT`

Launcher behavior:
- `run_qemu_capture.sh` defaults `CONSUMER_SCRIPT` to `capture_consumer_qemu.sh`

Implication:
- consumer membership is part of the active flow, but the exact selection mechanism is indirect rather than explicit in Python

### Raw Retention Exists But Is Not Active By Current Config
Active config:
- `rawRetention.enabled` is `false` in `config_qemu_upc.json`

Implication:
- raw dump retention, raw matrix construction, and raw metrics should be documented as a dormant branch, not as active default behavior

### Optional Step Files Exist But Are Not Always Active
Repository contents:
- `steps_phase_blocks.txt`
- `steps_cycle_repetition.txt`
- `steps_transition_stress.txt`

Controller behavior:
- these files matter only when `STEPS_FILE` points to one of them

Implication:
- they belong in the booklet as optional scenario definitions, not as mandatory pipeline stages

### External Metrics Internals Are Not Reconstructed Here
Observed behavior:
- `capture_consumer_qemu.sh` and `offline_step_metrics.py` call into external modules such as `coherence_temp_spec_stability.streaming_metrics`, `stability_validator`, and `plv_calcolator`

Implication:
- this booklet should describe those modules as dependencies and interface points, but not claim detailed formulas or internal algorithms unless another repository source directly supports them

## Explicit Exclusions
The following files exist in the repository but are excluded from the active architecture unless a future phase explicitly activates them:

- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_user_raw.sh`
- `VM_sampler/VM_Capture_QEMU/cleanup_qemu_capture.sh`
- `VM_sampler/VM_Capture_QEMU/fix_line_endings_server.sh`
- `VM_sampler/VM_Capture_QEMU/RAW_CAPTURE_ALTERNATIVE.md`
- `VM_executables/run_files.sh`

## Documentation Guidance
When extending this booklet later:

- prefer the active code path over generic README prose
- mark config-disabled branches as conditional, not active
- distinguish host-side paths from guest-side assumptions
- keep repository-wide historical branches out of scope unless the active controller begins to depend on them
