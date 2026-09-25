# Пример предсказания

## Подготовка окружения

Установите зависимости один раз:

```shell
make setup
```

Сборка Cython-расширения также выполняется автоматически командой `make predict`. Если нужно собрать его отдельно, выполните:

```shell
make build
```

## Предсказание с локальными весами

Веса не входят в репозиторий. Если они уже находятся в каталоге `weights/`, запустите:

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

## Пользовательские пути

Для собственных каталогов и весов используйте явные пути:

```shell
make predict \
  DATA_DIR=/data/images \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1" \
  RGB_SUFFIX=png \
  GPUS=1
```

По умолчанию входные файлы должны иметь имена `*_RGB.png`. Для изменения расширения укажите `RGB_SUFFIX`.

## Запуск через Slurm

Для отправки задания на кластер используйте `make predict-cluster`. Если веса уже находятся в локальном каталоге `weights/`, выполните:

```shell
make predict-cluster \
  DATA_DIR=images2 \
  OUTPUT_DIR=submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1" \
  CLUSTER_PARTITION=hiperf \
  CLUSTER_NODE=tesla-a100 \
  CLUSTER_GPUS=8
```

Команда использует Slurm и запускает prediction с общей файловой системы. Параметры `CLUSTER_GPUS`, `CLUSTER_NODE`, `CLUSTER_PARTITION`, `CLUSTER_CPUS`, `CLUSTER_MEMORY` и `CLUSTER_TIME` можно переопределить непосредственно в команде.

## Прямой запуск обёртки

Те же параметры можно передать непосредственно в `predict.sh`:

```shell
./predict.sh \
  --weights weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_1_last_1 \
  --input-dir /data/images \
  --output-dir /data/submission \
  --rgb-suffix png \
  --gpus 1 \
  --tile-size 2048
```

Дополнительно можно настроить `BATCH_SIZE`, `DOWNSAMPLE`, `UNIT`, `COMPRESS` и `MASTER_PORT` через Makefile или переменные окружения.
