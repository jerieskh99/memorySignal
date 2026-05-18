# Safety model — VM_executables_phase2

This document defines the safety guarantees of the Phase 2 workload generators
and how they are enforced.

## Scope

Every test in `VM_executables_phase2/` is a controlled workload generator. The
goal is to drive the guest with reproducible memory- and IO-access patterns
while the host sampler captures the memory signal. Nothing in this folder is
malware. Nothing here performs real encryption, persistence, evasion, network
communication, privilege escalation, credential access, spreading, or any
destructive action.

The strongest guarantees are needed for the SECURITY-LIKE family
(`sandbox_ransom_*`, `sandbox_scanner_metadata`). These tests are
**behavioral-pattern simulators only**: they mimic the *workload shape* of
file-processing pipelines (discover → read → transform → write → rename)
without any of the operational characteristics of real malicious software.

## Behavioral pattern mimicked (whitelist)

- Staged directory / file discovery inside the sandbox.
- Metadata enumeration (`stat`, `readdir`) inside the sandbox.
- Repeated open/read/transform/write/rename per file inside the sandbox.
- Bursty (batched) and slow-burn (paced) execution variants.
- CPU + IO mixture from a toy *reversible* transformation on generated files.
- Phase structure that produces a recognizable mechanism-aligned segment trace.

## Behaviors that are NOT mimicked or implemented (blacklist)

- Persistence of any kind (no autostart, no daemonization, no service
  registration, no scheduled tasks, no startup-folder writes).
- Evasion: no anti-debugging, anti-VM, anti-sandbox, anti-analysis logic.
- Stealth: no hidden files, no process hiding, no rootkit techniques.
- Privilege escalation: no setuid/setgid use, no kernel exploitation, no
  capability requests beyond what the launching user has.
- Credential access: no reading of `/etc/shadow`, password stores, keychains,
  browser profiles, ssh keys, or any user secrets.
- Spreading: no network sockets at all, no email, no IPC outside the test
  process, no copying of itself.
- Real cryptography: the only transform is XOR with a fixed 32-byte key
  derived deterministically from the run seed. There is no symmetric or
  asymmetric encryption, no KDF, no nonces, no IV chaining, no real ciphers.
- Destructive behavior: no `unlink` outside the sandbox, no overwriting of
  real files, no deletion of original copies on disk.
- Real malware logic: no deployment, no C2 protocol, no domain generation, no
  beaconing, no campaign identifiers.

## Sandbox guarantees

For every SECURITY-LIKE test (and for the file-touching MEM/APP-REALISTIC
tests where applicable), the following checks run before any file is created
or modified:

1. **Absolute path required.** The supplied `--sandbox-dir` (or backing path
   for `mem_mmap_traversal_v2`) must start with `/`.
2. **No `..` components.** Any textual `..` segment in the supplied path
   causes the test to abort.
3. **Realpath under an approved root.** The canonical resolution of the
   sandbox path must start with one of: `/tmp/`, `/var/tmp/`, or an explicit
   `--safe-root PATH` supplied by the operator. Resolution uses `realpath()`
   so symlink escapes are caught.
4. **No symlink targets in cleanup.** The cleanup walker calls `lstat` on
   every entry, refuses symlinks pointing outside the sandbox, and returns
   non-zero if any escaping symlink is found.
5. **Hard caps.** Before file generation begins, the test verifies:
   - `num_files <= P2_SANDBOX_MAX_FILES` (5000)
   - `num_files * file_size <= P2_SANDBOX_MAX_BYTES` (5 GiB)
   - `duration <= P2_SANDBOX_MAX_DURATION_S` (600 s)
   Exceeding any cap aborts the test before any file is touched.
6. **No following of symlinks on creation.** All `open()` calls during file
   creation pass `O_NOFOLLOW`. If a path is a symlink, the call fails and the
   test aborts.
7. **No exec.** The tests never call `exec*` or `system()`. (The Python
   methodology script `mp_phase_boundary_inference` does spawn a child via
   `subprocess.run`, but only the C binary path the operator supplies on the
   CLI; the child is itself a Phase 2 binary subject to the same rules.)
8. **No network.** None of the tests open any AF_INET / AF_INET6 / AF_UNIX
   socket. Grepping for `socket(` / `bind(` / `connect(` / `sendto(` /
   `import socket` returns no matches.

## Dry-run support

Every test accepts `--dry-run`. In this mode the test:

- parses all flags,
- validates the sandbox path,
- writes a metadata JSON with `"status": "dry_run"`,
- emits a log line describing what *would* have been done,
- exits without creating any files or modifying any state.

Operators should use `--dry-run` once with each new parameter set before
committing to a real run.

## Cleanup

- `--cleanup` is **on by default** for SECURITY-LIKE tests.
- The cleanup routine refuses any path that does not pass the validator.
- It also refuses to remove the sandbox if any escaping symlink is detected.
- For APP-REALISTIC tests, `--cleanup` removes the generated input, output,
  WAL, SHM, and DB files unless `--keep-db` (or equivalent) is set.

`scripts/cleanup_sandboxes.sh` provides an out-of-band cleanup pass for
operator use: it removes any leftover `/tmp/phase2_sandbox_*` or
`/var/tmp/phase2_sandbox_*` directories, refusing anything outside those
roots.

## What an auditor can verify

```sh
# 1. No network code.
grep -RIn -E 'socket\(|connect\(|bind\(|sendto\(|sendmsg\(|recvfrom\(|recvmsg\(' \
    common/ mem/ security_like_safe/ app_realistic/ methodology/ \
    | grep -v 'README\|\.md' || echo "OK: no network syscalls in workload code"

grep -RIn -E '^\s*import\s+(socket|urllib|http|requests|httpx)' \
    common/ app_realistic/ methodology/ || echo "OK: no Python network imports"

# 2. No exec.
grep -RIn -E 'execve\(|system\(|popen\(' \
    common/ mem/ security_like_safe/ app_realistic/ || echo "OK: no exec syscalls"

# 3. Sandbox validator is called before file generation.
grep -RIn 'p2_sandbox_validate' security_like_safe/ common/

# 4. Hard caps are referenced.
grep -RIn 'P2_SANDBOX_MAX_' security_like_safe/ common/

# 5. Cleanup routine refuses unvalidated paths.
grep -RIn 'refuse to remove' common/phase2_common.h
```

## Substitutions vs. original task brief

The task brief listed behaviors that the SECURITY-LIKE tests should NOT
mimic. We honor that list verbatim:

- Persistence: not implemented.
- Evasion / stealth: not implemented.
- Privilege escalation: not implemented; tests run as the invoking user.
- Credential access: not implemented; no reads of `~/.ssh`, `/etc/shadow`,
  keychains, password managers.
- Spreading: not implemented; no network or file copies outside the sandbox.
- Real encryption of user files: not implemented; the only transform is a
  reversible XOR on generated sandbox files.
- Destructive behavior: not implemented; only sandbox-internal files are
  modified or removed.
- Real malware deployment logic: not implemented; no payload delivery,
  loader, beacon, or campaign identifier.

If a future variant introduces, say, an extra discovery rule, it must
preserve every entry in this list.
