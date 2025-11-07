#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

SCRIPT_PATH="/mnt/aix7804/multivsr/inference_single.py"
CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/model.pth"
VISUAL_CKPT_PATH="/mnt/aix7804/multivsr/checkpoints/feature_extractor.pth"
SRC_DIR="/mnt/aix7804/multivsr/dataset/data/metadata/mtedx"

mapfile -t FILES < <(find "${SRC_DIR}" -type f -name '*.mp4' | sort)

for filepath in "${FILES[@]}"; do
    rel_path="${filepath#${SRC_DIR}/}"
    echo "Running inference: ${rel_path}"
    python "${SCRIPT_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --visual_encoder_ckpt_path "${VISUAL_CKPT_PATH}" \
        --fpath "${filepath}" "$@"
done
