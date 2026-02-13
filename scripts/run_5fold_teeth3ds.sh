#!/bin/bash
# ============================================================
# 5-Fold Cross-Validation for Teeth3DS with Sonata Fine-tuning
# ============================================================
#
# Usage:
#   bash scripts/run_5fold_teeth3ds.sh [SONATA_WEIGHT_PATH]
#
# Example:
#   bash scripts/run_5fold_teeth3ds.sh exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
#
# If no weight path is given, training starts from scratch.
# ============================================================

set -e

WEIGHT=${1:-""}
NUM_GPU=${2:-1}
CONFIG="semseg-sonata-teeth3ds-ft"
DATASET="teeth3ds"

for FOLD in 0 1 2 3 4; do
    EXP_NAME="${CONFIG}-fold${FOLD}"
    echo "========================================"
    echo "  Starting Fold ${FOLD} / 4"
    echo "  Experiment: ${EXP_NAME}"
    echo "========================================"

    # Build train split: all folds except current
    TRAIN_SPLIT="["
    for i in 0 1 2 3 4; do
        if [ "$i" != "$FOLD" ]; then
            if [ "$TRAIN_SPLIT" != "[" ]; then
                TRAIN_SPLIT="${TRAIN_SPLIT},"
            fi
            TRAIN_SPLIT="${TRAIN_SPLIT}\"fold_${i}\""
        fi
    done
    TRAIN_SPLIT="${TRAIN_SPLIT}]"

    VAL_SPLIT="fold_${FOLD}"

    if [ -n "$WEIGHT" ]; then
        sh scripts/train.sh -p python -g ${NUM_GPU} -d ${DATASET} -c ${CONFIG} -n ${EXP_NAME} \
            -w ${WEIGHT} \
            --options "data.train.split=${TRAIN_SPLIT}" "data.val.split=${VAL_SPLIT}" "data.test.split=${VAL_SPLIT}"
    else
        sh scripts/train.sh -p python -g ${NUM_GPU} -d ${DATASET} -c ${CONFIG} -n ${EXP_NAME} \
            --options "data.train.split=${TRAIN_SPLIT}" "data.val.split=${VAL_SPLIT}" "data.test.split=${VAL_SPLIT}"
    fi

    echo "Fold ${FOLD} finished."
    echo ""
done

echo "========================================"
echo "  All 5 folds completed!"
echo "========================================"
