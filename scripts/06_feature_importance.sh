#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory

# options
FEAT_IMPORTANCE_DIR="${ROOT_DIR}/feature-importance"
INTERESTINGNESS_DIR="${ROOT_DIR}/interestingness"
INTERESTINGNESS_FILE="${INTERESTINGNESS_DIR}/interestingness_pd.pkl.gz"
DATA_COLLECTION_DIR="${ROOT_DIR}/interaction-data"
INT_DATA_FILE="${DATA_COLLECTION_DIR}/interaction_data.pkl.gz"
HIGHLIGHTS_DIR="${ROOT_DIR}/highlights"
HIGHLIGHTS_FILE="${HIGHLIGHTS_DIR}/cluster-overall/highlights.csv"
FEAT_IMPORTANCE_DIR="${ROOT_DIR}/feature-importance"
CLUSTERS_FILE="${ROOT_DIR}/clustering/clustering5/trace-clusters.csv"

ROLLOUT_COL="Trace ID"
CLUSTER_COL="Cluster"

SEED=17
IMG_FORMAT="pdf"
CLEAR=false # whether to clear output/results directory
VERBOSITY=1
PARALLEL=-1 # num processes (usually = available cpus)

# mongo config for hyperopt (optional)
USE_MONGO_DB=false
MONGO_PORT="1234"
MONGO_DB="localhost:${MONGO_PORT}/ixdrl"
MONGO_TEMP="${FEAT_IMPORTANCE_DIR}/_mongo_temp"

if $USE_MONGO_DB; then
  # launch mongo server in the background and store PID (MONGO_DB_PATH needs to be defined in the envrionment)
  mongod --dbpath ${MONGO_DB_PATH} --port $MONGO_PORT & #--directoryperdb --journal &
  MONGOD_PID=$!

  # clear database prior to optimization
  sleep "5s"
  echo "Deleting database..."
  mongo ${MONGO_DB} --eval "db.dropDatabase()" --verbose

  # launch hyperopt worker in the background and store PID
  hyperopt-mongo-worker --mongo=${MONGO_DB} --poll-interval=0.1 --workdir ${MONGO_TEMP} &
  HYPER_MONGO_PID=$!
fi

python -m ixdrl.bin.feature_importance \
  -i "${INTERESTINGNESS_FILE}" \
  -d "${INT_DATA_FILE}" \
  -hi "${HIGHLIGHTS_FILE}" \
  -o "${FEAT_IMPORTANCE_DIR}" \
  --rollout-col="${ROLLOUT_COL}" \
  --cluster-col="${CLUSTER_COL}" \
  --processes=${PARALLEL} \
  --seed=${SEED} \
  --format "${IMG_FORMAT}" \
  --verbosity ${VERBOSITY} \
  --clear ${CLEAR} #\
#  --clusters="${CLUSTERS_FILE}"

if $USE_MONGO_DB; then
  # kill mongo processes
  echo "Killing processes..."
  kill ${HYPER_MONGO_PID}
  kill ${MONGOD_PID}

  # remove temp dir
  rm -rf ${MONGO_TEMP}
fi

# save pip packages to file
echo "Saving pip packages..."
pip freeze >"$FEAT_IMPORTANCE_DIR/packages.txt"
