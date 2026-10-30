#!/bin/bash
# Rent a vast.ai GPU, deploy the code, download the needed models, start a job, and arm
# a local watcher that copies results back and destroys the instance when done.
#
#   [JOB=run_sniff.sh] [DONE=JOB_DONE] [SCAN_BATCH=2048] \
#     scripts/vast_launch.sh <offer_id> <name> <job args...>
#
# Job args name model keys from models.py (for run_jobs.sh: kind:model, e.g. detect:pls).
# Requires: vastai CLI logged in, an SSH key at ~/.ssh/id_ed25519.pub, and a Hugging Face
# login with access to the gated models (the token is piped to the instance via a 0600
# file and deleted after downloading, never put on a command line).
# Safety: the instance stops itself after 4 h; the watcher gives up after 4 h, copies what
# exists, and destroys the instance only if the copy succeeded (otherwise it stops it).
set -u
OFFER=$1 NAME=$2; shift 2
JOB=${JOB:-run_sniff.sh} DONE=${DONE:-JOB_DONE}
MLIST=$(for j in "$@"; do echo "${j##*:}"; done | sort -u | tr "\n" " ")
cd "$(dirname "$0")/.."
ID=$(vastai create instance $OFFER --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime --disk 60 --ssh --direct \
     --label sniff-$NAME --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['new_contract'] if d.get('success') else '')")
[ -z "$ID" ] && { echo "$NAME: rent failed"; exit 1; }
echo "$NAME: instance $ID"
vastai attach ssh $ID "$(cat ~/.ssh/id_ed25519.pub)" > /dev/null
for i in $(seq 80); do
  S=$(vastai show instances --raw | python3 -c "import json,sys; d=[x for x in json.load(sys.stdin) if x['id']==$ID]; print(d[0].get('actual_status') if d else 'gone')")
  [ "$S" = running ] && break; sleep 15
done
[ "$S" = running ] || { echo "$NAME: never started, destroying"; vastai destroy instance $ID -y; exit 1; }
read H P <<< "$(vastai ssh-url $ID | sed -E 's#ssh://([^:]+):([0-9]+)#\1 \2#')"
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=15 -p $P"
for i in $(seq 10); do $SSH $H true 2>/dev/null && break; sleep 10; done
rsync -az -e "$SSH" common.py models.py scan.py search.py sniff.py functional.py scripts/$JOB data ${EXTRA:-} \
  $H:/workspace/exp/ 2>/dev/null
python3 -c "from huggingface_hub import get_token; print(get_token())" | $SSH $H "umask 077; cat > /root/.hf_token
pip install -q 'transformers==5.20.2' numpy scipy safetensors huggingface_hub vastai -U brotli zstandard > /dev/null 2>&1
cd /workspace/exp && HF_TOKEN=\$(cat /root/.hf_token) python -c \"
from huggingface_hub import snapshot_download
from models import MODELS
for k in ['base'] + '$MLIST'.split():
    for r in {MODELS[k]['repo'], MODELS[k]['base']}:
        snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
print('downloads OK')
\"; rm -f /root/.hf_token
export SCAN_BATCH=${SCAN_BATCH:-2048}; nohup ./$JOB $* > results_log.txt 2>&1 &
eval \$(tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(API_KEY|ID)=' | sed 's/^/export /')
nohup sh -c \"sleep 14400; vastai stop instance \\\$CONTAINER_ID --api-key \\\$CONTAINER_API_KEY\" > /workspace/selfstop.log 2>&1 &
echo started" 2>&1 | grep -v "Welcome\|Have fun"
mkdir -p logs
DONE=$DONE OUT=results_gpu/$NAME nohup caffeinate -i scripts/vast_watch.sh $ID $H $P 4 > logs/watch_$NAME.log 2>&1 &
echo "$NAME: watcher armed ($ID $H $P)"
