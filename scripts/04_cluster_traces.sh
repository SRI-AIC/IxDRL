#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/.." || exit
clear

# options
ROOT_DIR="output/breakout-dist-dqn"  #  relative to parent directory
DATA_COLLECTION_DIR="${ROOT_DIR}/interaction-data"
INT_DATA_FILE="${DATA_COLLECTION_DIR}/interaction_data.pkl.gz"
VIDEOS_DIR="${DATA_COLLECTION_DIR}/videos"
INTERESTINGNESS_DIR="${ROOT_DIR}/interestingness"
INTERESTINGNESS_FILE="${INTERESTINGNESS_DIR}/interestingness_pd.pkl.gz"
TRACE_CLUSTERING_DIR="${ROOT_DIR}/clustering"
BY_CLUSTER_DIR="${INTERESTINGNESS_DIR}/by-cluster"

DATA_DIR="${TRACE_CLUSTERING_DIR}/data"
TRACES_FILE="${DATA_DIR}/all-traces-train.pkl.gz"
TRACE_ID_COL="Rollout"
TIMESTEP_COL="Timestep"
START_COL=2 # rollout id, timestep -> then features
SPLIT=1     # use all data

EMBEDDINGS_DIR="${TRACE_CLUSTERING_DIR}/embeddings"
EMBEDDINGS_FILE="${EMBEDDINGS_DIR}/embeddings.pkl.gz"
EMBED_ALG="mean"
FILTER_CONST=true

DISTANCES_DIR="${TRACE_CLUSTERING_DIR}/distances"
DISTANCES_FILE="${DISTANCES_DIR}/trace-distances.npz"
DIST_METRIC="euclidean"

N_CLUSTERS=5
CLUSTERS_DIR="${TRACE_CLUSTERING_DIR}/clustering${N_CLUSTERS}"
EVAL_CLUSTERS=15  # maximum number of cluster to perform internal/external evaluation
LINKAGE="complete"
DISTANCE_THRESH=0.04
CLUSTERS_FILE="${CLUSTERS_DIR}/trace-clusters.csv"

VIDEO_AMOUNT=10
OUTPUT_VIDEOS_DIR="${TRACE_CLUSTERING_DIR}/videos${N_CLUSTERS}"
ROLLOUT_COL="Trace ID"
CLUSTER_COL="Cluster"

SEED=17
IMG_FORMAT="pdf"
CLEAR=false # whether to clear output/results directory
VERBOSITY=1
PARALLEL=-1 # num processes (usually = available cpus)

# script options
SPLIT_DATA=true
GET_EMBEDDINGS=true
COMPUTE_DISTANCES=true
CLUSTER_TRACES=true
COPY_VIDEOS=true
ANALYZE_BY_CLUSTER=true

# INSTALL DEPENDENCIES
pip install git+https://github.com/SRI-AIC/trace-clustering.git

# SPLIT DATA
if $SPLIT_DATA; then
  echo "========================================"
  echo "Preparing interestingness data for clustering..."
  echo "========================================"
  python -m trace_clustering.bin.split_data \
    -i "${INTERESTINGNESS_FILE}" \
    -o "${DATA_DIR}" \
    -s ${SPLIT} \
    --trace-col "${TRACE_ID_COL}" \
    --timestep-col "${TIMESTEP_COL}" \
    --start-col "${START_COL}" \
    --seed ${SEED} \
    -v ${VERBOSITY} \
    -c ${CLEAR}
fi

# GET EMBEDDINGS
if $GET_EMBEDDINGS; then
  echo "========================================"
  echo "Extracting embedding using '${EMBED_ALG}'..."
  echo "========================================"
  python -m trace_clustering.bin.get_embeddings \
    -i "${TRACES_FILE}" \
    -o "${EMBEDDINGS_DIR}" \
    -ea ${EMBED_ALG} \
    --filter-constant ${FILTER_CONST} \
    --processes ${PARALLEL} \
    -v ${VERBOSITY} \
    -c ${CLEAR}
fi

# COMPUTE TRACES DISTANCES
if $COMPUTE_DISTANCES; then
  echo "========================================"
  echo "Computing trace pairwise distances using '${DIST_METRIC}'..."
  echo "========================================"
  python -m trace_clustering.bin.get_distances \
    -e "${EMBEDDINGS_FILE}" \
    -o "${DISTANCES_DIR}" \
    -dm "${DIST_METRIC}" \
    --processes ${PARALLEL} \
    -f ${IMG_FORMAT} \
    -v ${VERBOSITY} \
    -c ${CLEAR}
fi

# CLUSTER TRACES
if $CLUSTER_TRACES; then
  echo "========================================"
  echo "Clustering traces using HAC with '${LINKAGE}' linkage and ${N_CLUSTERS} clusters..."
  echo "========================================"
  python -m trace_clustering.bin.cluster_traces \
    -d "${DISTANCES_FILE}" \
    -t "${TRACES_FILE}" \
    -e "${EMBEDDINGS_FILE}" \
    -o "${CLUSTERS_DIR}" \
    -f "${IMG_FORMAT}" \
    -l "${LINKAGE}" \
    -dt ${DISTANCE_THRESH} \
    -n ${N_CLUSTERS} \
    -ec ${EVAL_CLUSTERS} \
    --processes ${PARALLEL} \
    -v ${VERBOSITY} \
    -c ${CLEAR}
fi

# COPY VIDEOS FOR EACH CLUSTER
if $COPY_VIDEOS; then
  echo "========================================"
  echo "Copying videos for clustering with ${N_CLUSTERS} clusters..."
  echo "========================================"
  python -m ixdrl.bin.copy_videos \
    --input=${INT_DATA_FILE} \
    --clusters=${CLUSTERS_FILE} \
    --rollout-col="${ROLLOUT_COL}" \
    --cluster-col="${CLUSTER_COL}" \
    --output="${OUTPUT_VIDEOS_DIR}" \
    --amount=${VIDEO_AMOUNT} \
    --seed=${SEED} \
    --video-dir="${VIDEOS_DIR}" \
    --verbosity=${VERBOSITY} \
    --clear=${CLEAR}
fi

# ANALYZE INTERESTINGNESS PER CLUSTER
if $ANALYZE_BY_CLUSTER; then
  echo "========================================"
  echo "Analyzing interestingness for ${N_CLUSTERS} clusters..."
  echo "========================================"
  python -m ixdrl.bin.analyze_by_cluster \
    -i "${INTERESTINGNESS_DIR}" \
    -o "${BY_CLUSTER_DIR}" \
    --clusters="${CLUSTERS_FILE}" \
    --rollout-col="${ROLLOUT_COL}" \
    --cluster-col="${CLUSTER_COL}" \
    --format "${IMG_FORMAT}" \
    --verbosity ${VERBOSITY} \
    --clear ${CLEAR}
fi