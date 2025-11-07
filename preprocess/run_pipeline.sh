#!/bin/bash

SCRIPT_DIR="/mnt/aix7804/multivsr/preprocess"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

SRC_DIR="/mnt/aix7804/e-mvsr/data/mTED_data/fr-fr/data/test/trimmed_videos"
RESULT_DIR="/mnt/aix7804/multivsr/preprocess/mTEDx_results"
SCRIPT_PATH="${SCRIPT_DIR}/run_pipeline.py"

mkdir -p "${RESULT_DIR}"


cd "${SCRIPT_DIR}"

for filepath in "${SRC_DIR}"/*.mp4; do
    found_any=true
    filename=$(basename "${filepath}")
    reference="${filename%.mp4}.mp4"

    echo "Processing: ${filename} -> ${reference}"

    python "${SCRIPT_PATH}" \
        --videofile "${filepath}" \
        --reference "${reference}" \
        --data_dir "${RESULT_DIR}"
done


