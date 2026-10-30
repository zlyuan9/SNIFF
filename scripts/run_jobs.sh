#!/bin/bash
# Run a list of jobs: ./run_jobs.sh detect:<model> ... functional:<model> ...
# detect = sniff.py; functional = functional.py on results/detect_<model>.json.
# Writes results/JOB_DONE at the end.
cd "$(dirname "$0")"
mkdir -p results
export SCAN_BATCH=${SCAN_BATCH:-2048} SCAN_TF32=1 HF_HUB_OFFLINE=1
for job in "$@"; do
  IFS=: read kind m <<< "$job"
  echo "=== $job"
  if [ "$kind" = detect ]; then python sniff.py "$m"; else python functional.py "$m"; fi 2>&1 \
    | grep -v -i "warn\|Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "$job" >> results/FAILED.txt
done
touch results/JOB_DONE
