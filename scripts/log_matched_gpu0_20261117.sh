#!/usr/bin/env bash
set -euo pipefail
cd /root/adablock
out=results/matched_cobs_jssa_full_20261117/full100/gpu0_telemetry.csv
if [[ ! -e "$out" ]]; then
  echo 'timestamp,index,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu' > "$out"
fi
while kill -0 8519 2>/dev/null; do
  nvidia-smi -i 0 --query-gpu=timestamp,index,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$out" || true
  sleep 30
done
