#!/bin/bash

SCRIPT_DIR="/mnt/aix7804/multivsr/dataset"
VIDEOS_FOLDER="/mnt/aix7804/multivsr/dataset/data/videos"
CLIPS_ROOT="/mnt/aix7804/multivsr/dataset/data/metadata/multivsr"
TEMP_DIR="/mnt/aix7804/multivsr/dataset/data/temp"
SCRIPT_PATH="${SCRIPT_DIR}/preprocess_single.py"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=4,5,6,7

mkdir -p "${TEMP_DIR}"

cd "${SCRIPT_DIR}" || exit 1

for clip in "${CLIPS_ROOT}"/*/*.pckl; do
    video_id=$(basename "$(dirname "${clip}")")
    video_file="${VIDEOS_FOLDER}/${video_id}.mp4"

    if [[ ! -f "${video_file}" ]]; then
        echo "Skip: missing video for ${video_id}"
        continue
    fi

    echo "Processing: ${clip} -> ${video_file}"

    python "${SCRIPT_PATH}" \
        --video_file "${video_file}" \
        --track_file "${clip}" \
        --temp_dir "${TEMP_DIR}" \
        --frame_rate 25
done
