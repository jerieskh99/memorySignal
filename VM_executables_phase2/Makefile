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
	mem/mem_mmap_traversal_v2.c

SEC_SRCS := \
	security_like_safe/sandbox_ransom_seq.c \
	security_like_safe/sandbox_ransom_batched.c \
	security_like_safe/sandbox_ransom_slowburn.c \
	security_like_safe/sandbox_ransom_selective.c \
	security_like_safe/sandbox_scanner_metadata.c

APP_SRCS := \
	app_realistic/app_hashtable_intensive_v2.c

C_SRCS  := $(MEM_SRCS) $(SEC_SRCS) $(APP_SRCS)
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
