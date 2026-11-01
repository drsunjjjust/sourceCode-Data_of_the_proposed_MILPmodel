#!/bin/bash
# Full heuristic study (v2), resumable: every step is skipped when its flag exists; the CSV writers
# resume interrupted steps.  Relaunch with experiments/ensure.sh after a machine restart.
cd "$(dirname "$0")/.."
R=results
step () {  # step <flag> <command...>
  local flag=$1; shift
  if [ -f "$R/$flag" ]; then return; fi
  echo "[$(date +%H:%M:%S)] start $flag" >> $R/pipeline.log
  "$@" >> $R/pipeline.log 2>&1 && touch "$R/$flag" && echo "[$(date +%H:%M:%S)] done $flag" >> $R/pipeline.log
}
step calib.flag     python3 experiments/run_heuristics.py calib --workers 2
step calibsum.flag  python3 analysis/calib.py
step protoE.flag    python3 experiments/run_heuristics.py protoE --workers 2 --params $R/calibrated_params.json
step robust.flag    python3 experiments/run_heuristics.py robust --workers 2 --params $R/calibrated_params.json
step ablation.flag  python3 experiments/run_heuristics.py ablation --workers 2 --params $R/calibrated_params.json
step len.flag       python3 experiments/run_heuristics.py len --workers 2 --params $R/calibrated_params.json
step usage.flag     python3 experiments/run_usage.py --seeds 2 --params $R/calibrated_params.json
step protoT.flag    python3 experiments/run_heuristics.py protoT --workers 2
step sens.flag      python3 experiments/run_heuristics.py sens --workers 2 --params $R/calibrated_params.json
step qcap.flag      python3 experiments/run_heuristics.py qcap --workers 2 --params $R/calibrated_params.json
touch $R/all_done.flag
echo "[$(date +%H:%M:%S)] ALL DONE" >> $R/pipeline.log
