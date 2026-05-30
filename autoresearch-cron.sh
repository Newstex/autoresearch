#!/bin/bash
# autoresearch-cron.sh
# Off-peak GPU utilization cron wrapper
# Runs karpathy autoresearch loop ONLY during off-peak hours (UTC 23-07)
# and ONLY if GPU utilization is under 40%.
# Uses maximum niceness (lowest CPU priority) so it's first killed on OOM.

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
WORKSPACE="/home/newstex/workspace/autoresearch"
CODEX_BIN="/home/newstex/.hermes/node/bin/codex"
LOCKFILE="/tmp/autoresearch-cron.lock"
PIDFILE="/tmp/autoresearch-cron.pid"
GPU_THRESHOLD=40
OFFPEAK_START=23   # UTC hour when off-peak begins
OFFPEAK_END=7      # UTC hour when off-peak ends (stop before peak)
MAX_RUN_SECONDS=14400  # Max 4 hours per session (safety limit)

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
LOG_DIR="/home/newstex/.hermes/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/autoresearch-cron.log"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"
}

# ──────────────────────────────────────────────────────────────
# Time check: only run during off-peak hours (UTC 23-07)
# ──────────────────────────────────────────────────────────────
current_hour=$(date -u +%H)
current_hour_dec=$(date -u +%k)

if [ "$current_hour_dec" -ge "$OFFPEAK_START" ] || [ "$current_hour_dec" -lt "$OFFPEAK_END" ]; then
    log "Off-peak hour (UTC $current_hour) — proceeding"
else
    log "Peak hour (UTC $current_hour) — skipping. Off-peak is UTC $OFFPEAK_START-$OFFPEAK_END"
    exit 0
fi

# ──────────────────────────────────────────────────────────────
# GPU utilization check
# ──────────────────────────────────────────────────────────────
# Note: Blackwell GB10 GPU-Util reports compute mode % (always ~94% even idle).
# Use memory utilization instead — 0% means no active compute workload.
GPU_MEM_UTIL=$(nvidia-smi -q 2>/dev/null | grep -A4 "Utilization" | grep "Memory" | awk '{print $3}' | sed 's/%//' | head -1)

if [ -z "$GPU_MEM_UTIL" ]; then
    GPU_MEM_UTIL=0
fi

if [ "$GPU_MEM_UTIL" -lt "$GPU_THRESHOLD" ]; then
    log "GPU memory utilization $GPU_MEM_UTIL% < $GPU_THRESHOLD% — proceeding"
else
    log "GPU memory utilization $GPU_MEM_UTIL% >= $GPU_THRESHOLD% — skipping"
    exit 0
fi

# ──────────────────────────────────────────────────────────────
# Lock check — don't run multiple instances
# ──────────────────────────────────────────────────────────────
if [ -f "$LOCKFILE" ]; then
    # Check if lock is stale (process no longer running)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        log "Autoresearch already running (PID $(cat "$PIDFILE")) — skipping"
        exit 0
    else
        log "Stale lock found — removing"
        rm -f "$LOCKFILE" "$PIDFILE"
    fi
fi

# ──────────────────────────────────────────────────────────────
# Data readiness check
# ──────────────────────────────────────────────────────────────
if [ ! -d "$HOME/.cache/autoresearch/data" ] || [ ! -d "$HOME/.cache/autoresearch/tokenizer" ]; then
    log "Data not prepared — running prepare.py"
    cd "$WORKSPACE"
    uv run prepare.py 2>&1 >> "$LOG"
fi

# ──────────────────────────────────────────────────────────────
# Create lock
# ──────────────────────────────────────────────────────────────
echo $$ > "$PIDFILE"
touch "$LOCKFILE"
log "Starting autoresearch loop (PID $$)"

# ──────────────────────────────────────────────────────────────
# Run the autoresearch Python loop with maximum niceness
# ──────────────────────────────────────────────────────────────
#
# Strategy: launch the Python loop controller which uses Hermes's
# own LLM provider to make decisions and iterate experiments.
# Maximum niceness ensures it yields CPU to any other process.
#
# The Python script handles time limits internally (stops at 07:00).
# ──────────────────────────────────────────────────────────────

cd "$WORKSPACE"
nice -n 19 \
    ionice --class idle \
    timeout 21600 \
    python3 autoresearch-loop.py \
    2>&1 >> "$LOG"

EXIT_CODE=$?
log "Codex exited with code $EXIT_CODE"

# ──────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────
rm -f "$LOCKFILE" "$PIDFILE"
log "Autoresearch cron complete (exit $EXIT_CODE)"
exit $EXIT_CODE
