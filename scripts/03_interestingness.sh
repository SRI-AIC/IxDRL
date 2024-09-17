#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory
INT_DATA_FILE="${ROOT_DIR}/interaction-data/interaction_data.pkl.gz"
INTERESTINGNESS_DIR="${ROOT_DIR}/interestingness"
DERIVATIVE_ACC=2

IMG_FORMAT="pdf"
CLEAR=false # whether to clear output/results directory
VERBOSITY="info"
PARALLEL=-1 # num processes (usually = available cpus)

python -m ixdrl.bin.analyze \
  -i "${INT_DATA_FILE}" \
  -o ${INTERESTINGNESS_DIR} \
  --derivative_accuracy ${DERIVATIVE_ACC} \
  --img-format "${IMG_FORMAT}" \
  --clear ${CLEAR} \
  --verbosity ${VERBOSITY} \
  --processes ${PARALLEL}

# save pip packages to file
echo "Saving pip packages..."
pip freeze >"$INTERESTINGNESS_DIR/packages.txt"
