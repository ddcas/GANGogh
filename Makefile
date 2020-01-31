ENV?=staging
PROJECT_ID?=syb-$(ENV)-ai
ROOT_BUCKET?=$(PROJECT_ID)-language-detection
TAG_NAME?=covergen
JOB_DATE?=$(shell date +'%Y%m%d_%H%M%S')
JOB_NAME?=$(TAG_NAME)_nn_$(JOB_DATE)
REGION?=us-central1
# REGION?=us-central1-c

JOB_DIR?=gs://$(ROOT_BUCKET)-training-jobs/
# DATA_DIR?=gs://$(ROOT_BUCKET)-training-album-covers/
DATA_DIR?=gs://$(ROOT_BUCKET)-training-audio-data/0_utils/

JOB_DIR_LOCAL?=output/
DATA_DIR_LOCAL?=datasets/

NUM_FILES?=15000

train-cloud:
	gcloud ml-engine jobs submit training $(JOB_NAME) \
        --project $(PROJECT_ID) \
        --package-path trainer \
        --module-name trainer.task \
        --config trainer/cloudml.yaml \
        --job-dir "$(JOB_DIR)/$(JOB_NAME)" \
        --region $(REGION) \
        -- \
        --data-dir $(DATA_DIR) \
        --num-files $(NUM_FILES)

hypersearch:
	gcloud ml-engine jobs submit training $(JOB_NAME) \
        --project $(PROJECT_ID) \
        --package-path trainer \
        --module-name trainer.task \
        --config trainer/hypersearch.yaml \
        --job-dir "$(JOB_DIR)/$(JOB_NAME)" \
        --region $(REGION) \
        -- \
        --one-vs-all $(ONE_VS_ALL) \
        --svm $(SVM) \
        --debug $(DEBUG) \
        --num-files $(NUM_FILES) \
        --data-dir $(DATA_DIR) \
        --set-lang $(SET_LANG) \
        --num-epochs $(NUM_EPOCHS) \
        --oversampling $(OVERSAMPLING)

train-local-notreally:
	rm -rf $(JOB_DIR_LOCAL)
	gcloud ml-engine local train \
        --package-path trainer \
        --module-name trainer.task \
        --job-dir "$(JOB_DIR_LOCAL)" \
        -- \
        --data-dir $(DATA_DIR) \
        --num-files $(NUM_FILES)

train-local:
	rm -rf $(JOB_DIR_LOCAL)
	gcloud ml-engine local train \
        --package-path trainer \
        --module-name trainer.task \
        --job-dir "$(JOB_DIR_LOCAL)" \
        -- \
        --data-dir $(DATA_DIR_LOCAL) \
        --num-files $(NUM_FILES)
