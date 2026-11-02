#!/bin/bash
# ============================================================================
# sample_resources_mac.sh   (macOS, Apple Silicon)
#
# Samples CPU, memory, the llama-server process and GPU use/power, and appends one
# CSV row per sample. Run it in its OWN Terminal window, start it before the batch,
# Ctrl+C to stop.
#
#   sudo ./sample_resources_mac.sh [output.csv] [seconds] [process-name]
#
# Run with sudo to fill the GPU columns (powermetrics needs root). Without sudo
# everything else still works and the GPU columns are empty.
# First time:  chmod +x sample_resources_mac.sh
#
# Columns:
#   sample_time_utc   UTC (matches started_utc in each run's manifest.json)
#   cpu_total_pct     whole-machine CPU use (100 - idle)
#   mem_used_mb       active + wired + compressed memory, MB
#   llama_cpu_pct     llama-server CPU (100 = one core busy; can exceed 100)
#   llama_rss_bytes   llama-server resident memory, bytes
#   gpu_util_pct      GPU active residency (needs sudo)
#   gpu_power_w       GPU power draw in watts (needs sudo)
# Memory is unified on Apple Silicon, so there is no separate VRAM figure; the model
# shows up in llama_rss_bytes / mem_used_mb.
# ============================================================================
OUT="${1:-resource_samples.csv}"
INTERVAL="${2:-5}"
PROC="${3:-llama-server}"

[ -s "$OUT" ] || echo "sample_time_utc,cpu_total_pct,mem_used_mb,llama_cpu_pct,llama_rss_bytes,gpu_util_pct,gpu_power_w" > "$OUT"
if [ "$(id -u)" -eq 0 ]; then echo "Running as root: GPU columns will be filled."; else echo "Not root: GPU columns will be empty (use sudo for them)."; fi
echo "Sampling every ~${INTERVAL}s to $OUT (process: $PROC). Ctrl+C to stop."

while true; do
  T=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # CPU: second sample of top is the real interval average
  CPU=$(top -l 2 -n 0 -s 1 | grep "CPU usage" | tail -1 | awk '{gsub("%","",$7); printf "%.1f", 100-$7}')

  # Memory (active + wired + compressed), MB
  MEM=$(vm_stat | awk '
    /page size of/ {ps=$8}
    /Pages active/ {a=$3+0}
    /Pages wired down/ {w=$4+0}
    /Pages occupied by compressor/ {c=$5+0}
    END {printf "%.0f", (a+w+c)*ps/1048576}')

  # llama-server process (empty when not running)
  PID=$(pgrep -x "$PROC" | head -1)
  PCPU=""; PRSS=""
  if [ -n "$PID" ]; then
    read PCPU RSSKB <<< "$(ps -o %cpu=,rss= -p "$PID")"
    [ -n "$RSSKB" ] && PRSS=$((RSSKB*1024))
  fi

  # GPU (root only)
  GU=""; GP=""
  if [ "$(id -u)" -eq 0 ]; then
    PM=$(powermetrics --samplers gpu_power -i 1000 -n 1 2>/dev/null)
    GU=$(echo "$PM" | awk '/GPU HW active residency/ {gsub("%","",$5); print $5; exit}')
    GP=$(echo "$PM" | awk '/^GPU Power/ {printf "%.2f", $3/1000; exit}')
  fi

  echo "$T,$CPU,$MEM,$PCPU,$PRSS,$GU,$GP" >> "$OUT"
  sleep "$INTERVAL"
done
