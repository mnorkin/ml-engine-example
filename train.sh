#!/usr/bin/env bash

PROJECT_ID="wixpre"
JOB_NAME="test_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://$PROJECT_ID-ml/$JOB_NAME"
TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"
DATA_DIR="gs://$PROJECT_ID-ml/data"

echo "PROJECT: $PROJECT_ID, JOB_NAME: $JOB_NAME, JOB_DIR: $JOB_DIR"

echo "Copy data"

#gsutil -m cp -r data ${DATA_DIR}

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --job-dir ${JOB_DIR} \
    --package-path ${TRAINER_PACKAGE_PATH} \
    --module-name ${MAIN_TRAINER_MODULE} \
    --region us-central1 \
    -- \
    --data-dir=${DATA_DIR}

gcloud ml-engine jobs stream-logs ${JOB_NAME}
