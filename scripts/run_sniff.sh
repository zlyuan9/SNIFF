#!/bin/bash
# Run SNIFF on one or more models: ./run_sniff.sh <model> ...
# Starts with a quick smoke test on the first model. Writes results/JOB_DONE at the end.
cd "$(dirname "$0")"
mkdir -p results
export SCAN_BATCH=${SCAN_BATCH:-2048} SCAN_TF32=1 HF_HUB_OFFLINE=1
echo "=== smoke test $1"
DETECT_QUICK=1 python sniff.py "$1" 2>&1 | grep -v -i "warn\|Loading weights"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "smoke test FAILED" >> results/FAILED.txt; touch results/JOB_DONE; exit 1; fi
for model in "$@"; do
  echo "=== $model"
  python sniff.py "$model" 2>&1 | grep -v -i "warn\|Loading weights"
  [ "${PIPESTATUS[0]}" -eq 0 ] || echo "$model" >> results/FAILED.txt
done
touch results/JOB_DONE
