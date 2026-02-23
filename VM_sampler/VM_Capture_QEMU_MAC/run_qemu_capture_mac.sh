#!/usr/bin/env bash
# Start QEMU/macOS capture producer and consumer. Producer captures VM memory via QEMU monitor;
# consumer runs delta + streaming metrics on snapshots and then deletes them.
# Both run alongside each other (consumer processes jobs while producer keeps capturing).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCER_SCRIPT="${PRODUCER_SCRIPT:-$ROOT/capture_producer_qemu_mac.sh}"
CONSUMER_SCRIPT="${CONSUMER_SCRIPT:-$ROOT/capture_consumer_qemu_mac.sh}"
CONFIG="${CONFIG:-$ROOT/config_qemu_mac.json}"
BACKGROUND="${BACKGROUND:-0}"

if [[ ! -f "$PRODUCER_SCRIPT" ]]; then
  echo "Producer not found: $PRODUCER_SCRIPT"
  exit 1
fi
if [[ ! -f "$CONSUMER_SCRIPT" ]]; then
  echo "Consumer not found: $CONSUMER_SCRIPT"
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG (copy config_qemu_mac.json.example to config_qemu_mac.json and edit)"
  exit 1
fi

export CONFIG

if [[ "${BACKGROUND}" == "1" || "${BACKGROUND}" == "true" ]]; then
  echo "Starting producer and consumer in background (same terminal, nohup)."
  nohup bash "$PRODUCER_SCRIPT" >> "$ROOT/producer.log" 2>&1 &
  PROD_PID=$!
  nohup bash "$CONSUMER_SCRIPT" >> "$ROOT/consumer.log" 2>&1 &
  CONS_PID=$!
  echo "Producer PID: $PROD_PID (log: $ROOT/producer.log)"
  echo "Consumer PID: $CONS_PID (log: $ROOT/consumer.log)"
  echo "Root: $ROOT"
  echo "To stop: kill $PROD_PID $CONS_PID"
else
  echo "Starting producer and consumer in separate terminals."
  echo "Root: $ROOT"
  echo "Producer: $PRODUCER_SCRIPT"
  echo "Consumer: $CONSUMER_SCRIPT"
  echo "Run in two terminals:"
  echo "  terminal 1: cd $ROOT && CONFIG=$CONFIG bash $PRODUCER_SCRIPT"
  echo "  terminal 2: cd $ROOT && CONFIG=$CONFIG bash $CONSUMER_SCRIPT"
  echo "Or set BACKGROUND=1 to run in background: BACKGROUND=1 $0"
fi
