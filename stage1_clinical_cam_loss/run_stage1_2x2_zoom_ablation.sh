#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/path/to/env/bin/python"
PROJECT_ROOT="/path/to/project/mnn"
TRAINER="$PROJECT_ROOT/stage1_effnet_ce_train_full_image_clinical_attention.py"
EXPERIMENT_ROOT="$PROJECT_ROOT/stage1_2x2_zoom_ablation"
# Training seed, read from the SEED environment variable (no seed value is stored in the code).
SEED="${SEED:?set SEED}"

mkdir -p "$EXPERIMENT_ROOT"
export MPLCONFIGDIR="/tmp/stage1_2x2_zoom_mpl"
export OMP_NUM_THREADS=4
# The cluster environment's OpenCV links against libGL, supplied by base Conda.
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

run_cell() {
    local cell_name="$1"
    local attention_weight="$2"
    local suppression_prob="$3"
    local output_root="$EXPERIMENT_ROOT/$cell_name"
    local completed_run

    mkdir -p "$output_root"
    completed_run="$(find "$output_root" -mindepth 3 -maxdepth 3 -type f -path '*/reports/final_summary.json' -print -quit 2>/dev/null || true)"
    if [[ -n "$completed_run" ]]; then
        echo "SKIP completed cell: $cell_name ($completed_run)"
        return 0
    fi

    echo "START cell=$cell_name attention_weight=$attention_weight suppression_prob=$suppression_prob"
    "$PYTHON_BIN" "$TRAINER" \
        --output-root "$output_root" \
        --attention-loss-weight "$attention_weight" \
        --attention-warmup-epochs 5 \
        --artifact-suppression-prob "$suppression_prob" \
        --zoom-min 0.90 \
        --zoom-max 1.10 \
        --seed "$SEED" \
        --epochs 50 \
        --patience 12 \
        --batch-size 8 \
        --num-workers 4 \
        2>&1 | tee "$output_root/training_console.log"
    echo "DONE cell=$cell_name"
}

# Factor A: CAM supervision (0 or 0.25); factor B: artifact suppression (0 or 0.50).
# Every cell uses detector-derived masks and identical full-image augmentation settings.
run_cell "ce_suppression_0"      0.00 0.00
run_cell "ce_suppression_50"     0.00 0.50
run_cell "cam_suppression_0"     0.25 0.00
run_cell "cam_suppression_50"    0.25 0.50

echo "All Stage-1 2x2 cells completed under $EXPERIMENT_ROOT"
