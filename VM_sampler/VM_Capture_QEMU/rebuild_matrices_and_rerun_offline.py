#!/usr/bin/env python3
"""
Rebuild per-test run_matrix .npy files from rotated delta frames, then rerun offline metrics.

Expected rotated layout:
  <rotated_root>/<test_name>/hamming/*.txt
  <rotated_root>/<test_name>/cosine/*.txt

Where <test_name> usually looks like: test6_mem_alloc_touch_pages

By default, this script reconstructs complex delta frames:
  D = hamming * exp(j * phase_scale * pi * cosine)

For each selected test, this script:
  1) Rebuilds run_matrix_<test_name>.npy in (pages, frames) layout.
  2) Calls offline_step_metrics.py using that matrix.

Baseline behavior:
  - Exactly one test is marked baseline (--is-baseline).
  - Default baseline is step number 1 (test1_*), or override with --baseline-test-name.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


TEST_NAME_RE = re.compile(r"^test(\d+)_.+$")
HAMMING_TS_RE = re.compile(r"memory_dump_hamming_results_par-(\d+)\.txt$")
COSINE_TS_RE = re.compile(r"memory_dump_cosine_results_par-(\d+)\.txt$")


def _step_index(test_name: str) -> int:
    match = TEST_NAME_RE.match(test_name)
    if not match:
        return 10**9
    return int(match.group(1))


def _sorted_frame_paths(frame_dir: Path, sort_mode: str) -> list[Path]:
    paths = [p for p in frame_dir.glob("*.txt") if p.is_file()]
    if sort_mode == "mtime":
        return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))
    return sorted(paths, key=lambda p: p.name)


def _extract_timestamp(path: Path, metric: str) -> str:
    regex = HAMMING_TS_RE if metric == "hamming" else COSINE_TS_RE
    match = regex.match(path.name)
    return match.group(1) if match else ""


def _pair_hamming_cosine_frames(
    hamming_paths: list[Path],
    cosine_paths: list[Path],
) -> list[tuple[Path, Path]]:
    h_map: dict[str, Path] = {}
    c_map: dict[str, Path] = {}

    for p in hamming_paths:
        ts = _extract_timestamp(p, "hamming")
        if ts:
            h_map[ts] = p
    for p in cosine_paths:
        ts = _extract_timestamp(p, "cosine")
        if ts:
            c_map[ts] = p

    if h_map and c_map and len(h_map) == len(hamming_paths) and len(c_map) == len(cosine_paths):
        common = sorted(set(h_map.keys()) & set(c_map.keys()), key=int)
        if len(common) != len(hamming_paths) or len(common) != len(cosine_paths):
            raise ValueError(
                "Could not pair all hamming/cosine frames by timestamp."
                f" hamming={len(hamming_paths)} cosine={len(cosine_paths)} common={len(common)}"
            )
        return [(h_map[ts], c_map[ts]) for ts in common]

    if len(hamming_paths) != len(cosine_paths):
        raise ValueError(
            "Could not pair frames by filename timestamp and frame counts differ:"
            f" hamming={len(hamming_paths)} cosine={len(cosine_paths)}"
        )

    print(
        "[REBUILD] WARNING: fallback pairing by sorted order (timestamp parse unavailable)."
    )
    return list(zip(hamming_paths, cosine_paths))


def discover_tests(rotated_root: Path, selected: set[str] | None) -> list[str]:
    tests = [p.name for p in rotated_root.iterdir() if p.is_dir()]
    if selected:
        tests = [t for t in tests if t in selected]
    tests.sort(key=lambda t: (_step_index(t), t))
    return tests


def rebuild_matrix_from_frames(frame_paths: list[Path], output_npy: Path) -> tuple[int, int]:
    if not frame_paths:
        raise ValueError("No frame files found.")

    output_npy.parent.mkdir(parents=True, exist_ok=True)

    first = np.loadtxt(frame_paths[0], dtype=np.float64).reshape(-1)
    num_pages = int(first.shape[0])
    num_frames = len(frame_paths)
    if num_pages == 0:
        raise ValueError(f"First frame is empty: {frame_paths[0]}")

    # Write directly to .npy without building an in-memory hstack.
    matrix = np.lib.format.open_memmap(
        str(output_npy),
        mode="w+",
        dtype=np.float64,
        shape=(num_pages, num_frames),
    )
    matrix[:, 0] = first

    for idx, frame_path in enumerate(frame_paths[1:], start=1):
        frame = np.loadtxt(frame_path, dtype=np.float64).reshape(-1)
        if frame.shape[0] != num_pages:
            raise ValueError(
                f"Page count mismatch in {frame_path}: {frame.shape[0]} != {num_pages}"
            )
        matrix[:, idx] = frame

    # Ensure flush.
    del matrix
    return num_pages, num_frames


def rebuild_complex_matrix_from_hamming_cosine(
    hamming_paths: list[Path],
    cosine_paths: list[Path],
    output_npy: Path,
    phase_scale: float,
) -> tuple[int, int]:
    if not hamming_paths or not cosine_paths:
        raise ValueError("No hamming/cosine frame files found.")

    pairs = _pair_hamming_cosine_frames(hamming_paths, cosine_paths)
    if not pairs:
        raise ValueError("No paired hamming/cosine frames found.")

    output_npy.parent.mkdir(parents=True, exist_ok=True)

    ham0 = np.loadtxt(pairs[0][0], dtype=np.float64).reshape(-1)
    cos0 = np.loadtxt(pairs[0][1], dtype=np.float64).reshape(-1)
    if ham0.shape[0] != cos0.shape[0]:
        raise ValueError(
            f"First paired frame page mismatch: {pairs[0][0]} vs {pairs[0][1]}"
        )
    num_pages = int(ham0.shape[0])
    num_frames = len(pairs)
    if num_pages == 0:
        raise ValueError("First paired frame is empty.")

    matrix = np.lib.format.open_memmap(
        str(output_npy),
        mode="w+",
        dtype=np.complex128,
        shape=(num_pages, num_frames),
    )
    matrix[:, 0] = ham0 * np.exp(1j * phase_scale * np.pi * cos0)

    for idx, (ham_path, cos_path) in enumerate(pairs[1:], start=1):
        ham = np.loadtxt(ham_path, dtype=np.float64).reshape(-1)
        cos = np.loadtxt(cos_path, dtype=np.float64).reshape(-1)
        if ham.shape[0] != num_pages or cos.shape[0] != num_pages:
            raise ValueError(
                f"Page count mismatch at frame {idx}: "
                f"hamming={ham.shape[0]} cosine={cos.shape[0]} expected={num_pages}"
            )
        matrix[:, idx] = ham * np.exp(1j * phase_scale * np.pi * cos)

    del matrix
    return num_pages, num_frames


def run_offline(
    python_bin: str,
    offline_script: Path,
    matrix_path: Path,
    test_name: str,
    output_root: Path,
    project_root: Path,
    baseline_dir: Path,
    window_size: int,
    step_size: int,
    is_baseline: bool,
) -> int:
    cmd = [
        python_bin,
        str(offline_script),
        "--matrix",
        str(matrix_path),
        "--step-name",
        test_name,
        "--output-root",
        str(output_root),
        "--project-root",
        str(project_root),
        "--baseline-dir",
        str(baseline_dir),
        "--window-size",
        str(window_size),
        "--step-size",
        str(step_size),
    ]
    if is_baseline:
        cmd.append("--is-baseline")

    print(f"[REBUILD] Running offline metrics for {test_name} ...")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rotated-root",
        required=True,
        help="Path to output_dir/rotated (contains test*/cosine|hamming folders).",
    )
    parser.add_argument(
        "--matrix-dir",
        required=True,
        help="Directory where rebuilt run_matrix_<test>.npy files will be written.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Offline output root passed to offline_step_metrics.py.",
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="memorySignal repo root passed to offline_step_metrics.py.",
    )
    parser.add_argument(
        "--representation",
        choices=("complex", "cosine", "hamming"),
        default="complex",
        help=(
            "Matrix representation to rebuild. "
            "'complex' uses BOTH hamming and cosine: "
            "D = hamming * exp(j * phase_scale * pi * cosine). "
            "Single-metric modes are kept for compatibility."
        ),
    )
    parser.add_argument(
        "--phase-scale",
        type=float,
        default=2.0,
        help="Scale in exp(j * phase_scale * pi * cosine). Default: 2.0",
    )
    parser.add_argument(
        "--baseline-step-number",
        type=int,
        default=1,
        help="Step number to mark as baseline when --baseline-test-name is not set.",
    )
    parser.add_argument(
        "--baseline-test-name",
        default="",
        help="Exact test folder name to use as baseline (overrides --baseline-step-number).",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=[],
        help="Optional explicit test folder names to process (e.g., test6_... test7_...).",
    )
    parser.add_argument(
        "--sort",
        choices=("name", "mtime"),
        default="name",
        help="Frame ordering policy inside each test folder (default: name).",
    )
    parser.add_argument(
        "--offline-script",
        default=str(Path(__file__).with_name("offline_step_metrics.py")),
        help="Path to offline_step_metrics.py.",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python executable for offline_step_metrics.py (default: python3).",
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining tests if one test fails.",
    )
    parser.add_argument(
        "--delete-matrix-after-success",
        action="store_true",
        help=(
            "Remove the rebuilt run_matrix_<test>.npy after offline_step_metrics.py "
            "exits successfully (rc=0) for that test."
        ),
    )
    args = parser.parse_args()

    rotated_root = Path(args.rotated_root)
    matrix_dir = Path(args.matrix_dir)
    output_root = Path(args.output_root)
    project_root = Path(args.project_root)
    offline_script = Path(args.offline_script)
    baseline_dir = output_root / "offline" / "baseline"

    if not rotated_root.is_dir():
        print(f"ERROR: rotated root not found: {rotated_root}", file=sys.stderr)
        return 1
    if not offline_script.is_file():
        print(f"ERROR: offline script not found: {offline_script}", file=sys.stderr)
        return 1

    selected = set(args.tests) if args.tests else None
    tests = discover_tests(rotated_root, selected=selected)
    if not tests:
        print("ERROR: no test folders found to process.", file=sys.stderr)
        return 1

    if args.baseline_test_name:
        baseline_test = args.baseline_test_name
    else:
        baseline_test = ""
        for test_name in tests:
            if _step_index(test_name) == args.baseline_step_number:
                baseline_test = test_name
                break

    if not baseline_test:
        print(
            "ERROR: could not resolve baseline test. "
            "Use --baseline-test-name explicitly or include the baseline step in --tests.",
            file=sys.stderr,
        )
        return 1
    if baseline_test not in tests:
        print(
            f"ERROR: baseline test '{baseline_test}' is not in selected tests.",
            file=sys.stderr,
        )
        return 1

    print(f"[REBUILD] Selected tests: {len(tests)}")
    print(f"[REBUILD] Baseline test: {baseline_test}")
    print(f"[REBUILD] Representation: {args.representation}")
    if args.representation == "complex":
        print(f"[REBUILD] Complex formula: hamming * exp(j * {args.phase_scale} * pi * cosine)")

    for test_name in tests:
        matrix_path = matrix_dir / f"run_matrix_{test_name}.npy"
        try:
            if args.representation == "complex":
                h_dir = rotated_root / test_name / "hamming"
                c_dir = rotated_root / test_name / "cosine"
                if not h_dir.is_dir() or not c_dir.is_dir():
                    raise ValueError(
                        f"Missing hamming/cosine dirs for {test_name}: {h_dir} {c_dir}"
                    )
                h_paths = _sorted_frame_paths(h_dir, args.sort)
                c_paths = _sorted_frame_paths(c_dir, args.sort)
                if not h_paths or not c_paths:
                    raise ValueError(
                        f"No hamming/cosine frame files for {test_name}."
                    )
                print(
                    f"[REBUILD] Rebuilding complex {matrix_path} from "
                    f"hamming={len(h_paths)} cosine={len(c_paths)} frame(s) ..."
                )
                num_pages, num_frames = rebuild_complex_matrix_from_hamming_cosine(
                    h_paths,
                    c_paths,
                    matrix_path,
                    phase_scale=args.phase_scale,
                )
            else:
                frame_dir = rotated_root / test_name / args.representation
                if not frame_dir.is_dir():
                    raise ValueError(f"Missing frame directory: {frame_dir}")
                frame_paths = _sorted_frame_paths(frame_dir, args.sort)
                if not frame_paths:
                    raise ValueError(f"No frame files under: {frame_dir}")
                print(
                    f"[REBUILD] Rebuilding {matrix_path} from {len(frame_paths)} frame(s) "
                    f"({frame_dir}) ..."
                )
                num_pages, num_frames = rebuild_matrix_from_frames(frame_paths, matrix_path)
        except Exception as exc:  # noqa: BLE001
            if args.continue_on_error:
                print(f"[REBUILD] WARNING: rebuild failed for {test_name}: {exc}")
                continue
            print(f"ERROR: rebuild failed for {test_name}: {exc}", file=sys.stderr)
            return 1

        print(
            f"[REBUILD] Matrix ready: {matrix_path} "
            f"shape=({num_pages}, {num_frames})"
        )

        rc = run_offline(
            python_bin=args.python_bin,
            offline_script=offline_script,
            matrix_path=matrix_path,
            test_name=test_name,
            output_root=output_root,
            project_root=project_root,
            baseline_dir=baseline_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            is_baseline=(test_name == baseline_test),
        )
        if rc != 0:
            if args.continue_on_error:
                print(f"[REBUILD] WARNING: offline metrics failed for {test_name} (rc={rc})")
                continue
            print(
                f"ERROR: offline metrics failed for {test_name} (rc={rc}).",
                file=sys.stderr,
            )
            return rc

        if args.delete_matrix_after_success:
            try:
                if matrix_path.is_file():
                    matrix_path.unlink()
                    print(f"[REBUILD] Deleted matrix after success: {matrix_path}")
                else:
                    print(
                        f"[REBUILD] WARNING: matrix not found for delete: {matrix_path}",
                        file=sys.stderr,
                    )
            except OSError as exc:
                print(
                    f"[REBUILD] WARNING: could not delete matrix {matrix_path}: {exc}",
                    file=sys.stderr,
                )

    print("[REBUILD] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

