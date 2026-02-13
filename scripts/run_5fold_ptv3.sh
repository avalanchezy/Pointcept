#!/bin/bash
# ============================================================
# 5-Fold Cross-Validation for Teeth3DS with Pure PTv3 (Scratch)
# ============================================================
#
# Usage:
#   bash scripts/run_5fold_ptv3.sh
#
# ============================================================

set -e

NUM_GPU=${1:-1}
CONFIG="semseg-ptv3-teeth3ds-scratch"
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

    # No weight argument (Training from scratch)
    sh scripts/train.sh -p python -g ${NUM_GPU} -d ${DATASET} -c ${CONFIG} -n ${EXP_NAME} \
        --options "data.train.split=${TRAIN_SPLIT}" "data.val.split=${VAL_SPLIT}" "data.test.split=${VAL_SPLIT}"

    echo "Fold ${FOLD} finished."
    echo ""
done

echo "========================================"
echo "  All 5 folds completed!"
echo "========================================"
