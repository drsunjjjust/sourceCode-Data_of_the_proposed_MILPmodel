#!/bin/bash
# relaunch the v2 pipeline if it is not running (after a machine restart)
cd "$(dirname "$0")/.."
if [ ! -f results/all_done.flag ] && ! pgrep -f "experiments/pipeline_v2.sh" > /dev/null; then
  (setsid nohup experiments/pipeline_v2.sh > results/pipeline.out 2>&1 < /dev/null &); echo "relaunched pipeline_v2"
fi
echo "checked $(date +%H:%M:%S)"; tail -3 results/pipeline.log 2>/dev/null
