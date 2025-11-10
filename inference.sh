#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

SCRIPT_PATH="/mnt/aix7804/multivsr/inference_multi.py"
CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/model.pth"
VISUAL_CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/feature_extractor.pth"
SRC_DIR="/mnt/aix7804/multivsr/dataset/data/metadata/mtedx"
METRICS_ROOT="/mnt/aix7804/multivsr/outputs/"
RUN_DATE="$(date +%Y-%m-%d)"
RUN_TIME="$(date +%H-%M-%S)"
LOG_DIR="${METRICS_ROOT%/}/${RUN_DATE}"
LOG_FILE="${LOG_DIR}/${RUN_TIME}.log"
mkdir -p "${LOG_DIR}"

python "${SCRIPT_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --visual_encoder_ckpt_path "${VISUAL_CKPT_PATH}" \
    --input_dir "${SRC_DIR}" \
    --lang_id "fr" "$@" \
    2>&1 | tee "${LOG_FILE}"

#################### Single file ####################
# python "${SCRIPT_PATH}" \
#     --ckpt_path "${CKPT_PATH}" \
#     --visual_encoder_ckpt_path "${VISUAL_CKPT_PATH}" \
#     --fpath "/path/to/video.mp4" \
#     --metrics_out "${METRICS_ROOT}" "$@"
