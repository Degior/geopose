# Prediction example

Set up the environment once:

```shell
make setup
```

Run prediction with the repository defaults:

```shell
make predict \
  DATA_DIR=images2 \
  OUTPUT_DIR=submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1" \
  RGB_SUFFIX=png \
  GPUS=1 \
  TILE_SIZE=2048 \
  COMPRESS=true
```

Run with custom paths:

```shell
make predict \
  DATA_DIR=/data/images \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="/data/weights/model-a.pt /data/weights/model-b.pt" \
  RGB_SUFFIX=png \
  GPUS=1
```

The input directory must contain files named `*_RGB.png` by default. The wrapper also accepts the options directly:

```shell
./predict.sh \
  --weights /data/weights/model-a.pt /data/weights/model-b.pt \
  --input-dir /data/images \
  --output-dir /data/submission \
  --rgb-suffix png \
  --gpus 1 \
  --tile-size 2048
```
