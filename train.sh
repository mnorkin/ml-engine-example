#!/usr/bin/env bash

PROJECT_ID="ml-finances"
JOB_NAME="test_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://$PROJECT_ID-ml/$JOB_NAME"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"
DATA_DIR="gs://$PROJECT_ID-ml/data"
MODEL_NAME="fin"

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
    --data-dir=${DATA_DIR}


gcloud ml-engine models create "$MODEL_NAME" \
  --regions us-central1
