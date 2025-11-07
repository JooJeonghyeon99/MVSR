#!/bin/bash

SCRIPT_DIR="/mnt/aix7804/multivsr/dataset"
PYWORK_DIR="/mnt/aix7804/multivsr/preprocess/mTEDx_results/pywork"
PYAVI_DIR="/mnt/aix7804/multivsr/preprocess/mTEDx_results/pyavi"
TEMP_DIR="/mnt/aix7804/multivsr/dataset/data/temp"
SCRIPT_PATH="${SCRIPT_DIR}/preprocess_single.py"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p "${TEMP_DIR}"

cd "${SCRIPT_DIR}" || exit 1

for track_file in "${PYWORK_DIR}"/*/tracks.pckl; do
    clip_id=$(basename "$(dirname "${track_file}")")
    video_file="${PYAVI_DIR}/${clip_id}/video.avi"

    if [[ ! -f "${video_file}" ]]; then
        echo "Skip: missing video for ${clip_id}"
        continue
    fi

    echo "Processing: ${track_file} -> ${video_file}"

    python "${SCRIPT_PATH}" \
        --video_file "${video_file}" \
        --track_file "${track_file}" \
        --temp_dir "${TEMP_DIR}" \
        --frame_rate 25
done
