#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
OUTPUT_DIR="output" # relative to parent directory
CONFIG_FILE="scripts/breakout_dist_dqn"
MOD_CONFIG_FILE="${CONFIG_FILE}_mod.yaml"
CONFIG_FILE="${CONFIG_FILE}.yaml"
CHECKPOINT_FREQ=3000

# edit storage_path option with output (needs to be complete path)
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" >/dev/null || exit; pwd -P)"
sed "s|# STORAGE_PATH|storage_path: \"${OUTPUT_DIR}\"|g" ${CONFIG_FILE} > ${MOD_CONFIG_FILE}

rllib train \
  file "${MOD_CONFIG_FILE}" \
  -vv \
  --ray-ui \
  --checkpoint-freq ${CHECKPOINT_FREQ} \
  --checkpoint-at-end
#  --resume

# save pip packages to file
pip freeze >"${OUTPUT_DIR}/packages.txt"
