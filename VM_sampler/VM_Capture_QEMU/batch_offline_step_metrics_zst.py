#!/usr/bin/env python3
"""
Batch wrapper for offline_step_metrics.py over .npy.zst matrix files.

For each .npy.zst file in --matrix-folder (processed one at a time):
  1. Decompress to a temporary .npy next to the compressed file.
  2. Run offline_step_metrics.py on that .npy.
  3. Delete the temporary .npy.
  4. Move to the next file.

Files are sorted numerically by the test number in their filename (test<N>).
The first sorted file is treated as the PLV baseline (--is-baseline).
All subsequent files load the baseline written by that first step.

The original .npy.zst files are never modified.
Decompression uses the system `zstd` CLI (must be on PATH).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_zstd() -> str:
    """Return the path to the zstd binary or raise if not found."""
    path = shutil.which("zstd")
    if path is None:
        raise RuntimeError(
            "zstd not found on PATH. Install it with your package manager "
            "(e.g. `brew install zstd` or `apt install zstd`) and retry."
        )
    return path


def _test_number(p: Path) -> tuple[int, str]:
    """Sort key: (test_number, name). Files without test<N> sort last."""
    m = re.search(r"test(\d+)", p.name, re.IGNORECASE)
    return (int(m.group(1)), "") if m else (10**9, p.name)


def _discover_files(matrix_folder: Path) -> list[Path]:
    """Return all .npy.zst files sorted by test number."""
    files = sorted(
        [p for p in matrix_folder.iterdir() if p.name.endswith(".npy.zst")],
        key=_test_number,
    )
    return files


def _step_name(matrix_zst: Path) -> str:
    """Derive the step label: strip .npy.zst from the filename.

    Example:
        run_matrix_test18_mem_alloc_touch_pages.npy.zst
        -> run_matrix_test18_mem_alloc_touch_pages
    """
    name = matrix_zst.name
    if name.endswith(".npy.zst"):
        return name[: -len(".npy.zst")]
    return name


def _tmp_npy_path(matrix_zst: Path) -> Path:
    """Temporary decompressed path: same dir, .npy.zst -> .npy."""
    return matrix_zst.with_suffix("")  # removes .zst, leaving .npy


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    matrix_zst: Path
    step: str
    is_baseline: bool
    decomp_ok: bool = False
    analysis_rc: int | None = None

    @property
    def ok(self) -> bool:
        return self.decomp_ok and self.analysis_rc == 0


def decompress(zstd_bin: str, matrix_zst: Path, tmp_npy: Path) -> bool:
    """Decompress matrix_zst to tmp_npy. Returns True on success."""
    cmd = [zstd_bin, "-d", str(matrix_zst), "-o", str(tmp_npy), "--force", "-q"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"  [DECOMP] ERROR rc={result.returncode}: {result.stderr.strip() or '(no stderr)'}"
            )
            return False
        return True
    except Exception as exc:
        print(f"  [DECOMP] ERROR: {exc}")
        return False


def run_offline(
    offline_script: Path,
    tmp_npy: Path,
    step: str,
    output_root: Path,
    project_root: Path,
    baseline_dir: Path,
    window_size: int,
    step_size: int,
    segments: int,
    min_windows_per_segment: int,
    is_baseline: bool,
) -> int:
    """Run offline_step_metrics.py on tmp_npy. Returns exit code."""
    cmd = [
        sys.executable,
        str(offline_script),
        "--matrix", str(tmp_npy),
        "--step-name", step,
        "--output-root", str(output_root),
        "--project-root", str(project_root),
        "--baseline-dir", str(baseline_dir),
        "--window-size", str(window_size),
        "--step-size", str(step_size),
        "--segments", str(segments),
        "--min-windows-per-segment", str(min_windows_per_segment),
    ]
    if is_baseline:
        cmd.append("--is-baseline")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as exc:
        print(f"  [OFFLINE] ERROR invoking offline_step_metrics.py: {exc}")
        return 1


def process_file(
    zstd_bin: str,
    offline_script: Path,
    matrix_zst: Path,
    is_baseline: bool,
    output_root: Path,
    project_root: Path,
    baseline_dir: Path,
    window_size: int,
    step_size: int,
    segments: int,
    min_windows_per_segment: int,
) -> FileResult:
    step = _step_name(matrix_zst)
    tmp_npy = _tmp_npy_path(matrix_zst)
    result = FileResult(matrix_zst=matrix_zst, step=step, is_baseline=is_baseline)

    tag = "[BASELINE]" if is_baseline else "[STEP]    "
    print(f"\n{tag} {matrix_zst.name}")
    print(f"  step-name : {step}")
    print(f"  tmp npy   : {tmp_npy.name}")

    # -- 1. Decompress --
    print(f"  Decompressing ...")
    result.decomp_ok = decompress(zstd_bin, matrix_zst, tmp_npy)
    if not result.decomp_ok:
        print(f"  Skipping offline analysis for {matrix_zst.name} (decompression failed).")
        return result

    # -- 2. Run offline analysis (cleanup happens regardless of outcome) --
    try:
        print(f"  Running offline_step_metrics.py ...")
        result.analysis_rc = run_offline(
            offline_script=offline_script,
            tmp_npy=tmp_npy,
            step=step,
            output_root=output_root,
            project_root=project_root,
            baseline_dir=baseline_dir,
            window_size=window_size,
            step_size=step_size,
            segments=segments,
            min_windows_per_segment=min_windows_per_segment,
            is_baseline=is_baseline,
        )
        if result.analysis_rc != 0:
            print(f"  [OFFLINE] WARNING: offline_step_metrics.py exited rc={result.analysis_rc}")
        else:
            print(f"  [OFFLINE] Done.")
    finally:
        # -- 3. Always delete temporary .npy --
        if tmp_npy.is_file():
            tmp_npy.unlink()
            print(f"  Deleted temporary {tmp_npy.name}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch wrapper: decompress .npy.zst files one at a time, run offline_step_metrics.py"
            " on each, delete the temporary .npy, and move to the next file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--matrix-folder", required=True, type=Path,
        help="Folder containing .npy.zst matrix files.",
    )
    parser.add_argument(
        "--folder-name", required=True,
        help="Label for this batch/output group (informational; printed in progress output).",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path,
        help="Root directory for offline outputs. Passed to offline_step_metrics.py.",
    )
    parser.add_argument(
        "--project-root", required=True, type=Path,
        help="Repo root; coherence_temp_spec_stability/ is loaded from here.",
    )
    parser.add_argument(
        "--baseline-dir", required=True, type=Path,
        help=(
            "Directory where baseline_plv.npy is written by the baseline step and read by all others."
            " All files in the batch share this directory."
        ),
    )
    parser.add_argument(
        "--window-size", type=int, default=128,
        help="Window length in frames for MSC/Cepstrum/PLV (default 128).",
    )
    parser.add_argument(
        "--step-size", type=int, default=64,
        help="Hop size in frames between windows (default 64).",
    )
    parser.add_argument(
        "--segments", type=int, default=1,
        help=(
            "Number of contiguous temporal segments for the optional segment-level pass"
            " (default 1 = no segmentation). Forwarded to offline_step_metrics.py."
        ),
    )
    parser.add_argument(
        "--min-windows-per-segment", type=int, default=50,
        help=(
            "Minimum windows per segment before emitting a warning (default 50)."
            " Forwarded to offline_step_metrics.py."
        ),
    )
    parser.add_argument(
        "--offline-script", type=Path,
        default=Path(__file__).with_name("offline_step_metrics.py"),
        help="Path to offline_step_metrics.py (default: same directory as this script).",
    )
    args = parser.parse_args()

    # -- Validate paths --
    if not args.matrix_folder.is_dir():
        print(f"ERROR: --matrix-folder not found: {args.matrix_folder}", file=sys.stderr)
        return 1
    if not args.offline_script.is_file():
        print(f"ERROR: offline script not found: {args.offline_script}", file=sys.stderr)
        return 1
    if args.segments < 1:
        print(f"ERROR: --segments must be >= 1, got {args.segments}", file=sys.stderr)
        return 1

    # -- Check zstd --
    try:
        zstd_bin = _find_zstd()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # -- Discover files --
    files = _discover_files(args.matrix_folder)
    if not files:
        print(f"WARNING: no .npy.zst files found in {args.matrix_folder}")
        return 0

    # -- Print processing order --
    print(f"Batch: {args.folder_name}")
    print(f"Files found: {len(files)}")
    print(f"Processing order:")
    for i, f in enumerate(files):
        label = " [BASELINE]" if i == 0 else ""
        print(f"  {i + 1:3d}. {f.name}{label}")
    print()

    # -- Process --
    results: list[FileResult] = []
    for i, matrix_zst in enumerate(files):
        r = process_file(
            zstd_bin=zstd_bin,
            offline_script=args.offline_script,
            matrix_zst=matrix_zst,
            is_baseline=(i == 0),
            output_root=args.output_root,
            project_root=args.project_root,
            baseline_dir=args.baseline_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            segments=args.segments,
            min_windows_per_segment=args.min_windows_per_segment,
        )
        results.append(r)

    # -- Summary --
    n_total = len(results)
    n_ok = sum(1 for r in results if r.ok)
    n_failed_decomp = sum(1 for r in results if not r.decomp_ok)
    n_failed_analysis = sum(
        1 for r in results if r.decomp_ok and r.analysis_rc is not None and r.analysis_rc != 0
    )

    print(f"\n{'=' * 50}")
    print(f"Batch summary: {args.folder_name}")
    print(f"  Total files    : {n_total}")
    print(f"  Succeeded      : {n_ok}")
    print(f"  Failed decomp  : {n_failed_decomp}")
    print(f"  Failed analysis: {n_failed_analysis}")

    if n_failed_decomp > 0 or n_failed_analysis > 0:
        print("\nFailed files:")
        for r in results:
            if not r.ok:
                reason = "decomp" if not r.decomp_ok else f"analysis rc={r.analysis_rc}"
                print(f"  {r.matrix_zst.name}  ({reason})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
