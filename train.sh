#!/usr/bin/env bash

PROJECT_ID="ml-finances"
VERSION=$(date +%Y%m%d_%H%M%S)
JOB_NAME="test_$VERSION"
JOB_DIR="gs://$PROJECT_ID-ml/$JOB_NAME"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"
DATA_DIR="gs://$PROJECT_ID-ml/data"
OUTPUT_DIR="gs://$PROJECT_ID-ml/output/$VERSION"
MODEL_NAME="fin"
MODEL_VERSION="v2"

echo "PROJECT: $PROJECT_ID, JOB_NAME: $JOB_NAME, JOB_DIR: $JOB_DIR"

echo "Copy data"

#gsutil -m cp -r data ${DATA_DIR}
# gcloud ml-engine local train \
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --stream-logs \
    --job-dir ${JOB_DIR} \
    --package-path ${TRAINER_PACKAGE_PATH} \
    --module-name ${MAIN_TRAINER_MODULE} \
    --region us-central1 \
    -- \
    --data-dir=${DATA_DIR} \
    --output-dir=${OUTPUT_DIR}

#gcloud ml-engine versions delete ${MODEL_VERSION} --model=${MODEL_NAME} -q --verbosity none
#gcloud ml-engine models delete ${MODEL_NAME} -q --verbosity none
#
#gcloud ml-engine models create ${MODEL_NAME} \
#  --regions us-central1
#
#gcloud ml-engine versions create ${MODEL_VERSION} \
#    --model "$MODEL_NAME" \
#    --origin "$OUTPUT_DIR"

