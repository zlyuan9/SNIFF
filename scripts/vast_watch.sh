#!/bin/sh
# Wait for a vast.ai job's DONE marker, copy results back, then destroy the instance.
#   [DONE=JOB_DONE] [OUT=results_gpu] scripts/vast_watch.sh <instance_id> <host> <port> <max_hours>
# Gives up at max_hours (copying whatever exists). Destroys only after a successful copy;
# if the copy fails it stops the instance instead (disk kept, GPU billing ends).
ID=$1 H=$2 P=$3 MAX=$4
cd "$(dirname "$0")/.."
OUT=${OUT:-results_gpu}
MARK=${DONE:-JOB_DONE}
DEADLINE=$(( $(date +%s) + MAX * 3600 ))
log() { echo "$(date '+%F %T') $*"; }
mkdir -p $OUT
while :; do
  STATE=$(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes -p $P $H \
    "ls /workspace/exp/results/$MARK 2>/dev/null" 2>/dev/null)
  case "$STATE" in *$MARK*) log "job finished"; break ;; esac
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "deadline reached"; break; }
  sleep 60
done
COPIED=0
for i in 1 2 3; do
  rsync -az -e "ssh -o ConnectTimeout=20 -p $P" "$H:/workspace/exp/results/" $OUT/ \
    && rsync -az -e "ssh -p $P" "$H:/workspace/exp/results_log.txt" $OUT/ && { COPIED=1; log "results copied to $OUT"; break; }
  log "copy attempt $i failed"; sleep 30
done
if [ $COPIED -eq 1 ]; then
  vastai destroy instance "$ID" -y && log "instance $ID destroyed"
else
  vastai stop instance "$ID" && log "COPY FAILED: instance $ID stopped, disk kept; restart it to retrieve results"
fi
