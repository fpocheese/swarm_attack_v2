#!/bin/bash
export PYTHONPATH="/home/uav/00gao_xueshu/muti_uav_attack:/home/uav/00gao_xueshu/muti_uav_attack/third_party/MACPO/MACPO"

echo "Waiting for process 1396394 to finish..."
while kill -0 1396394 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Training finished! Starting GIF evaluation immediately..."
/home/uav/anaconda3/envs/rlgpu/bin/python scripts/eval_fast_to_gifs.py > outputs/logs/eval_fast.log 2>&1
echo "[$(date)] Evaluation complete! GIFs generated in outputs/gifs/fast_run3_eval_10eps"
