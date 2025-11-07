#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

SCRIPT_PATH="/mnt/aix7804/multivsr/inference_folder.py"
CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/model.pth"
VISUAL_CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/feature_extractor.pth"
SRC_DIR="/mnt/aix7804/multivsr/dataset/data/metadata/mtedx"
METRICS_ROOT="/mnt/aix7804/multivsr/outputs/inference_metrics"

mkdir -p "${METRICS_ROOT}"

python "${SCRIPT_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --visual_encoder_ckpt_path "${VISUAL_CKPT_PATH}" \
    --input_dir "${SRC_DIR}" \
    --metrics_out "${METRICS_ROOT}" "$@"

# Single-file example (uncomment and supply your clip when needed):
# python "${SCRIPT_PATH}" \
#     --ckpt_path "${CKPT_PATH}" \
#     --visual_encoder_ckpt_path "${VISUAL_CKPT_PATH}" \
#     --fpath "/path/to/video.mp4" \
#     --metrics_out "${METRICS_ROOT}" "$@"