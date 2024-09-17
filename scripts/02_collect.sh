#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory
CHECKPOINT_DIR="${ROOT_DIR}/model" # path to policy checkpoints to analyze
DATA_COLLECTION_DIR="${ROOT_DIR}/interaction-data"
MODEL_NAME="DQN"

SEED=17
NUM_EPISODES=1000
NUM_WORKERS=1
NUM_ENVS_WORKER=20
NUM_GPUS=0

RESUME=true # tries to load previous data file, if does not exist, will collect data
CLEAR=false # whether to clear output/results directory
IMG_FORMAT="pdf"
VERBOSITY="info"
FPS=10 # the frames per second rate used to save the episode videos

CONFIG="{
\"seed\":${SEED},
\"num_workers\":${NUM_WORKERS},
\"num_envs_per_worker\":${NUM_ENVS_WORKER},
\"num_gpus\":${NUM_GPUS}
}"
CONFIG="${CONFIG//[$'\t\r\n ']/}" # remove whitespaces


#check latest checkpoint
CHECKPOINT=$(find "${CHECKPOINT_DIR}" -type d -name "checkpoint_*" -print | sort -r | head -n 1)
if [ -z "$CHECKPOINT" ]; then
  echo "No model checkpoint found!"
else
  echo "Loading from latest checkpoint: ${CHECKPOINT}"
  python -m ixdrl.bin.collect.rllib \
    "${CHECKPOINT}" \
    --output "${DATA_COLLECTION_DIR}" \
    --run "${MODEL_NAME}" \
    --config "${CONFIG}" \
    --episodes ${NUM_EPISODES} \
    --img-format "${IMG_FORMAT}" \
    --stats-only ${RESUME} \
    --clear ${CLEAR} \
    --verbosity ${VERBOSITY} \
    --fps ${FPS} \
    --render

  # save pip packages to file
  pip freeze >"${DATA_COLLECTION_DIR}/packages.txt"

fi