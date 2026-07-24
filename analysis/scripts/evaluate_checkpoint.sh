set -e

# allencahn2d / c5_na & c5_va_rewind
# burgers2d / c9_na & c7_va_rewind
# lambdaomega2d / c3_na & c7_va_rewind

EXPERIMENT_NAME=burgers2d
MODEL_NAME=mlp
EXPERIMENT_TAG=c7_va_rewind
DEVICE="cuda:4"

FOLDER="analysis/results/runs/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_${MODEL_NAME}_${EXPERIMENT_TAG}"

EVALUATE=false
MAKE_VIDEO=true
VIDEO_FILE_NAME=remade_video.mp4
VIDEO_GRID='{"nx":120,"ny":120,"nt":120}'

python -m analysis.evaluate_checkpoint \
  --experiment_name "$EXPERIMENT_NAME" \
  --model_name "$MODEL_NAME" \
  --folder_path "$FOLDER" \
  --device "$DEVICE" \
  --evaluate "$EVALUATE" \
  --make_video "$MAKE_VIDEO" \
  --video_file_name "$VIDEO_FILE_NAME" \
  --video_grid "$VIDEO_GRID"
