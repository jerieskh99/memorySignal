# Raw Physical Memory Capture: User-Space Alternative (No Privilege)

## Problem Summary

When the VM is managed under **qemu:///system**, libvirt runs the QEMU process under a confined service account (e.g. `libvirt-qemu`). As a result:

1. **pmemsave via QMP/HMP**  
   Writing memory dumps to a path in user space (e.g. `$HOME` or a project directory) fails with permission errors, because the QEMU process cannot write outside paths allowed by libvirt security policies.

2. **virsh dump --memory-only**  
   This produces ELF-format dumps and can write to a user-specified path, but the resulting files are owned by the QEMU/libvirt service account. Without elevated privileges you cannot reliably read, modify, or delete these files; processing them in a reproducible pipeline is therefore not feasible under strict user-space-only constraints.

Additionally, **ELF dumps are not flat raw physical memory**: they contain ELF headers and segment metadata. For **deterministic delta computation** and **mmap-based processing**, a **contiguous byte representation of guest physical memory** (flat raw) is required.

## Constraints (Strict)

The following are **out of scope** and must not be part of the solution:

- **No sudo / root** for capture or for post-processing dump files.
- **No changes to libvirt system configuration** (e.g. `qemu.conf`, `libvirtd.conf`).
- **No ACL, SELinux, or AppArmor policy changes** to allow the service account to write to user directories or to make dump files user-readable.

The solution must be **reproducible and defensible for academic experimentation**: a clear, deterministic acquisition methodology that can be described in a thesis or paper.

---

## Recommended Alternative: User-Run QEMU + Monitor pmemsave

**Idea:** For the **experimental capture workflow**, run the VM under your own user account (not under libvirt’s system instance). The QEMU process then runs as you, so it can write dump files into your directories. Use the QEMU monitor to pause the VM and run **pmemsave** to produce a **flat raw** guest physical memory image.

### Why This Satisfies the Requirements

| Requirement | How it is met |
|-------------|----------------|
| **Flat raw physical memory** | QEMU HMP command `pmemsave <addr> <size> <filename>` writes a raw byte stream of guest physical memory (no ELF/kdump headers). |
| **Snapshot consistency** | Send `stop` on the monitor before `pmemsave`, and `cont` after. The dump is taken while the VM is paused. |
| **User-space only** | No sudo, no libvirt system config, no policy changes. QEMU is started by you; all files are in your directories. |
| **Reproducible / defensible** | Methodology is explicit: “We used a user-started QEMU instance with a monitor socket in user-accessible storage. Memory was captured via the HMP command pmemsave while the VM was paused (stop → pmemsave → cont).” |

### Two Ways to Run QEMU as the User

1. **Direct QEMU (no libvirt)**  
   Start the VM with `qemu-system-x86_64` (or your arch) and pass a monitor socket in a user path, e.g.  
   `-monitor unix:/tmp/qemu-monitor.sock,server,nowait`  
   or a path under `$HOME`. All state and dumps stay in user space.

2. **libvirt qemu:///session**  
   Define and run the domain under the **session** connection (e.g. `virsh -c qemu:///session start <domain>`). The QEMU process runs as your user. You can then use **virsh -c qemu:///session** with the same domain; however, **virsh dump** still produces ELF and may still create files owned by the session’s QEMU process. For a **raw** dump without touching system libvirt, using **direct QEMU + monitor** is simpler and more predictable.

For a single, clear methodology we recommend **direct QEMU + monitor socket + pmemsave**.

---

## Methodology (Step-by-Step)

1. **Start the VM under your user**  
   Run QEMU with a monitor socket in a user-accessible path, e.g.:
   ```bash
   qemu-system-x86_64 -m 4096 \
     -monitor unix:/tmp/qemu-monitor.sock,server,nowait \
     -drive file=/path/to/disk.qcow2,if=virtio \
     ... other args ...
   ```
   Ensure the process is owned by you (e.g. you started it in your terminal or via a user-owned script).

2. **Determine guest RAM size**  
   You must know the guest physical RAM size used for the dump. If you started with `-m 4096`, that is 4096 MiB ⇒ `4096 * 1024 * 1024` bytes. Use this value as `<size>` in `pmemsave`. The producer script expects this as `ramSizeMb` in the config (see below).

3. **Capture sequence (single dump)**  
   - Connect to the monitor (e.g. `socat - UNIX-CONNECT:/tmp/qemu-monitor.sock`).
   - Send: `stop`  
     Wait until the VM is paused (optional: poll `info status` until it reports “paused”).
   - Send: `pmemsave 0 <size_in_bytes> /path/in/your/home/dump.raw`  
     Use physical address `0` and size = total guest RAM in bytes. The file is written by the QEMU process (which is you), so it will be in your directory and owned by you.
   - Send: `cont`  
   The file at `/path/in/your/home/dump.raw` is a **flat raw** image of guest physical memory, suitable for mmap and for deterministic delta computation (e.g. with `live_delta_calc`).

4. **Automation**  
   The script `capture_producer_qemu_user_raw.sh` in this directory implements the above loop: backpressure, pause → pmemsave → resume, and enqueue of (prev, curr) job pairs. It uses the same queue and consumer as the virsh-based pipeline; only the producer and the dump format (`.raw` instead of `.elf`) differ. The existing **consumer** and **live_delta_calc** work unchanged on the `.raw` files.

---

## Configuration (User-Raw Mode)

Use a config that includes:

- **qemuMonitorSocket**: path to the QEMU monitor Unix socket (e.g. `/tmp/qemu-monitor.sock`). Must match the `-monitor` argument used when starting QEMU.
- **ramSizeMb**: guest RAM size in MiB (e.g. `4096` for 4 GiB). Used to compute the byte size for `pmemsave 0 <size> <file>`.
- **imageDir**, **outputDir**, **queueDir**, **rustDeltaCalculationProgram**, etc.: same as in the virsh-based config. **imageDir** must be a path writable by your user (e.g. under `$HOME` or project directory).

Example (minimal) config for user-raw mode:

```json
{
  "qemuMonitorSocket": "/tmp/qemu-monitor.sock",
  "ramSizeMb": 4096,
  "imageDir": "/home/user/thesis/dumps",
  "outputDir": "/home/user/thesis/output",
  "queueDir": "/home/user/thesis/queue",
  "rustDeltaCalculationProgram": "/path/to/live_delta_calc",
  "intervalMsec": 500,
  "backpressure": { "maxPendingJobs": 20, "sleepOnBackpressureSeconds": 1 },
  "vmStatePolling": { "timeoutSeconds": 30, "pollIntervalMs": 200 }
}
```

No **domain** is needed for this producer (it does not use virsh). Optional sections (streaming, rawRetention) can be added as in the main pipeline; the consumer is shared.

---

## Summary

- **Under qemu:///system**: you cannot get a user-writable, flat raw dump without privilege or policy changes.
- **Under user-run QEMU** (direct or session): use the monitor to run `stop` → `pmemsave 0 <ramsize> <path>.raw` → `cont`. The result is a flat raw physical memory image, taken while the VM is paused, with no sudo and no libvirt/system policy changes, and with a clear, reproducible methodology for academic use.

The script `capture_producer_qemu_user_raw.sh` and this document provide the full alternative workflow for the VM_Capture_QEMU folder.
