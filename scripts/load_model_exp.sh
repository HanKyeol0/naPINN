# scripts/load_model_exp.sh

set -e

EXPERIMENT_NAME=burgers2d
MODEL_NAME=mlp
EXPERIMENT_TAG=1-1_naPINN_5-2
DEVICE=cuda

FOLDER="outputs/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}_${MODEL_NAME}_${EXPERIMENT_TAG}"

TRAIN=false
EVALUATE=false
MAKE_VIDEO=true
VIDEO_FILE_NAME=remade_video.mp4
VIDEO_GRID='{"nx":120,"ny":120,"nt":80}'

python -m pinnlab.load_model \
  --experiment_name $EXPERIMENT_NAME \
  --model_name $MODEL_NAME \
  --folder_path $FOLDER \
  --device $DEVICE \
  --train $TRAIN \
  --evaluate $EVALUATE \
  --make_video $MAKE_VIDEO \
  --video_file_name $VIDEO_FILE_NAME \
  --video_grid $VIDEO_GRID \