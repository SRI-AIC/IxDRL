#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory
INTERESTINGNESS_DIR="${ROOT_DIR}/interestingness"
DATA_COLLECTION_DIR="${ROOT_DIR}/interaction-data"
METADATA_FILE="${DATA_COLLECTION_DIR}/metadata.json"
HIGHLIGHTS_DIR="${ROOT_DIR}/highlights"
CLUSTERS_FILE="${ROOT_DIR}/clustering/clustering5/trace-clusters.csv"

ROLLOUT_COL="Trace ID"
CLUSTER_COL="Cluster"
MAX_HIGHLIGHTS=5 # maximum number of highlights to be recorded for each dimension
RECORD_STEPS=41  # num of timesteps to be recorded in each highlight video
FADE_RATIO=0.25  # ratio of frames to which apply a fade-in/out effect
IQR_MUL=1.5      # the IQR multiplier to determine outliers

IMG_FORMAT="pdf"
CLEAR=false # whether to clear output/results directory
VERBOSITY="info"

python -m ixdrl.bin.highlights \
  -i "${INTERESTINGNESS_DIR}" \
  -o "${HIGHLIGHTS_DIR}" \
  -m "${METADATA_FILE}" \
  --rollout-col="${ROLLOUT_COL}" \
  --cluster-col="${CLUSTER_COL}" \
  --max-highlights ${MAX_HIGHLIGHTS} \
  --record-timesteps ${RECORD_STEPS} \
  --fade-ratio ${FADE_RATIO} \
  --iqr-mul=${IQR_MUL} \
  --format "${IMG_FORMAT}" \
  --verbosity ${VERBOSITY} \
  --clear ${CLEAR} #\
#  --clusters="${CLUSTERS_FILE}" \

# save pip packages to file
echo "Saving pip packages..."
pip freeze >"${HIGHLIGHTS_DIR}/packages.txt"
