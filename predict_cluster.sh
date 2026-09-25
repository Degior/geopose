#!/bin/sh

NODE="tesla-a100"
SBATCH_CPU=""
SBATCH_CPU="-p hiperf -w ${NODE} --gres=gpu:8"

sbatch -n1 \
--cpus-per-task=10 \
--mem=40000 \
$SBATCH_CPU \
-t 20:00:00 \
--job-name=depth-eval \
--output=./logs/depth_eval_%j.log \
--error=./logs/depth_eval_%j.err \
\
--wrap='CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DATA_DIR=images2 \
    OUTPUT_DIR=submission \
      WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1" \
        RGB_SUFFIX=png \
	  GPUS=1 \
	    TILE_SIZE=2048 \
	      COMPRESS=true ./predict.sh'
