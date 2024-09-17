#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory
INTERESTINGNESS_DIR="${ROOT_DIR}/interestingness"
DATA_COLLECTION_DIR="${ROOT_DIR}/interaction-data"
TRAINING_INFO="${ROOT_DIR}/model/progress.csv"

streamlit run ixdrl/gui/inspector.py \
    --theme.base dark \
    -- \
    --interaction "${DATA_COLLECTION_DIR}" \
    --interestingness "${INTERESTINGNESS_DIR}" \
    --training "${TRAINING_INFO}"

# save pip packages to file
pip freeze >"${ROOT_DIR}/gui_packages.txt"
