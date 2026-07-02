# VM_executables_phase2 — Makefile
#
# Builds every C executable into bin/. Python and shell tests need no build.
#
# Usage:
#   make           # build everything
#   make smoke     # build + run --help + tiny dry-run for each test
#   make clean     # remove bin/ and intermediate object files
#
# Override compiler / flags from the command line, e.g.:
#   make CC=clang CFLAGS="-O2 -g -Wall -Wextra"

CC      ?= cc
CSTD    ?= -std=c11
CFLAGS  ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
CPPFLAGS?= -D_GNU_SOURCE
LDFLAGS ?=

BIN := bin

MEM_SRCS := \
	mem/mem_workingset_sweep_v2.c \
	mem/mem_writemag_sweep_v2.c \
	mem/mem_rmw_intensity_v2.c \
	mem/mem_pagefault_density_v2.c \
	mem/mem_mmap_traversal_v2.c \
	mem/mem_random_write_pages_v2.c \
	mem/mem_stride_sweep_large_v2.c

# Next-generation benign system-behaviour benchmark families (no network,
# no persistence; see docs/SAFETY_MODEL.md). THREAD_SRCS and MIXED_SRCS link
# with -pthread (see the per-directory pattern rules below).
CPU_SRCS := \
	cpu/cpu_hash_loop_v2.c \
	cpu/cpu_matrix_mult_v2.c \
	cpu/cpu_branch_random_v2.c

CACHE_SRCS := \
	cache/cache_hot_loop_v2.c \
	cache/cache_cold_scan_v2.c \
	cache/cache_stride_sweep_v2.c

IO_SRCS := \
	io/io_read_cache_hit_v2.c \
	io/io_direct_write_like_v2.c

THREAD_SRCS := \
	thread/thread_lock_contention_v2.c \
	thread/thread_producer_consumer_v2.c \
	thread/thread_parallel_alloc_v2.c

MIXED_SRCS := \
	mixed/mixed_mem_io_v2.c \
	mixed/mixed_cpu_mem_v2.c \
	mixed/mixed_cpu_io_v2.c

# Compute-kernel family (Berkeley dwarfs): structured-compute writers. Pure
# memory + compute, no -pthread. Sources are organised into per-dwarf
# subdirectories kernel/D<n>_<visible|quiet>_<name>/, where the directory name
# also marks whether that dwarf is write-VISIBLE or QUIET in the host signal.
KERNEL_SRCS := $(wildcard kernel/*/*.c)

SEC_SRCS := \
	security_like_safe/sandbox_ransom_seq.c \
	security_like_safe/sandbox_ransom_batched.c \
	security_like_safe/sandbox_ransom_slowburn.c \
	security_like_safe/sandbox_ransom_selective.c \
	security_like_safe/sandbox_scanner_metadata.c \
	security_like_safe/sandbox_stealth_microwrite.c \
	security_like_safe/sandbox_stealth_scattered.c \
	security_like_safe/sandbox_stealth_paced.c

APP_SRCS := \
	app_realistic/app_hashtable_intensive_v2.c

C_SRCS  := $(MEM_SRCS) $(SEC_SRCS) $(APP_SRCS) \
	$(CPU_SRCS) $(CACHE_SRCS) $(IO_SRCS) $(THREAD_SRCS) $(MIXED_SRCS) $(KERNEL_SRCS)
C_BINS  := $(patsubst %.c,$(BIN)/%,$(notdir $(C_SRCS)))

PY_TESTS := \
	app_realistic/app_sqlite_oltp_v2.py \
	app_realistic/app_sqlite_analytical_v2.py \
	app_realistic/app_compress_gzip_v2.py \
	app_realistic/app_decompress_gzip_v2.py \
	app_realistic/app_json_parse_v2.py \
	methodology/mp_phase_boundary_inference.py \
	methodology/mp_workingset_metric_linearity.py

.PHONY: all clean smoke list

all: $(C_BINS)
	@mkdir -p $(BIN)
	@chmod +x $(PY_TESTS) || true

# Pattern rule: build each .c into bin/<basename>.
$(BIN)/%: mem/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
$(BIN)/%: security_like_safe/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
$(BIN)/%: app_realistic/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
$(BIN)/%: cpu/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
$(BIN)/%: cache/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
$(BIN)/%: io/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS)
# Kernel dwarfs: sources live one level deeper (kernel/D<n>_.../*.c). vpath lets
# Make locate them by basename; -iquote kernel keeps the "../common" includes
# resolving from the deeper directory. This generic rule only fires for files
# vpath can find (the kernel subdirs), so the specific rules above are unaffected.
vpath %.c $(sort $(dir $(KERNEL_SRCS)))
$(BIN)/%: %.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -iquote kernel -o $@ $< $(LDFLAGS) -lm
# THREAD and MIXED use pthreads -> link with -pthread.
$(BIN)/%: thread/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -pthread
$(BIN)/%: mixed/%.c | $(BIN)
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) -pthread

$(BIN):
	@mkdir -p $(BIN)

clean:
	rm -rf $(BIN)

list:
	@echo "C executables:"
	@for b in $(C_BINS); do echo "  $$b"; done
	@echo "Python tests:"
	@for t in $(PY_TESTS); do echo "  $$t"; done

# Smoke test target — built by scripts/smoke_all.sh because the logic is more
# complex than a one-liner make recipe.
smoke: all
	@bash scripts/smoke_all.sh
