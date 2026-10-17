# Geopose Challenge — решение, занявшее первое место

## Обзор

Решение в основном основано на [статье CVPR EarthVision 2021](https://arxiv.org/abs/2105.08229) и [статье CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf).

## TLDR
### Подготовка
```shell
make all
```
### Веса
```shell
mkdir -p weights

wget -O weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 \
  "https://www.dropbox.com/s/0uplf7b5d502ul2/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2?dl=1"

wget -O weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 \
  "https://www.dropbox.com/s/54dhmlbdip23fif/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2?dl=1"
```
Эти два файла используются по умолчанию в `make predict` и `make predict-cluster`. После скачивания весов можно запускать предсказание без указания `WEIGHTS`.
### Предсказание
```shell
make predict \
  DATA_DIR=images2 \
  OUTPUT_DIR=submission \
  RGB_SUFFIX=png \
  GPUS=1 \
  TILE_SIZE=2048 \
  COMPRESS=true
```

## Требования

Проект запускается локально в виртуальном окружении Python. По умолчанию используется Python 3.9, а для вычислений на GPU необходимы совместимые драйвер NVIDIA и CUDA.

Зависимости из `requirements.txt` закреплены для локального окружения проекта, включая `pandas` и `albumentations` для обучения. Установка `GDAL` и `pygdal` также требует системных библиотек GDAL. Makefile по умолчанию использует индекс пакетов PyTorch для CUDA 12.4; при необходимости его можно изменить через `PIP_EXTRA_INDEX_URL`.

## Установка

Создайте виртуальное окружение и установите зависимости:

```shell
make venv
make install
```

`make setup` выполняет эти два шага одной командой. `make all` дополнительно собирает Cython-расширение:

```shell
make setup
make build
```

Если в системе используется другой Python, передайте его путь через `PYTHON`, например `make venv PYTHON=python3.10`.

## Предсказание на больших изображениях

Каталог с входными данными должен содержать изображения с именами `*_RGB.png` по умолчанию. В выходной каталог для каждого изображения сохраняются файлы `_AGL.tif` и `_VFLOW.json`; при включённом сжатии также создаётся отдельный каталог с конвертированными и сжатыми результатами.

Перед запуском скачайте веса из раздела TLDR. Makefile по умолчанию использует `weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2` и `weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2`. Их также можно указать явно через `WEIGHTS`:

```shell
make predict \
  DATA_DIR=/data/images \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2" \
  RGB_SUFFIX=png \
  GPUS=1
```

По умолчанию `make predict` также принимает `BATCH_SIZE`, `DOWNSAMPLE`, `UNIT`, `TILE_SIZE`, `COMPRESS` и `MASTER_PORT`. Например, `TILE_SIZE=1024` уменьшает потребление памяти GPU, а `COMPRESS=false` отключает постобработку и упаковку.

Те же параметры можно передать непосредственно в `predict.sh`:

```shell
./predict.sh \
  --model-path weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 \
  --dataset-dir /data/images \
  --predictions-dir /data/submission \
  --rgb-suffix png \
  --gpus 1 \
  --tile-size 2048
```

## Предсказание на Slurm-кластере

Команда `make predict-cluster` собирает Cython-расширение, создаёт каталог логов и отправляет задание через `sbatch`. По умолчанию используются текущие параметры кластера: partition `hiperf`, узел `tesla-a100`, 1 GPU, 8 CPU и 40 ГБ памяти. Перед отправкой убедитесь, что `sbatch` доступен, а исходный код и виртуальное окружение находятся на общей файловой системе.

Если веса из TLDR уже находятся в локальном каталоге `weights/`, пример запуска выглядит так:

```shell
make predict-cluster \
  DATA_DIR=images2 \
  OUTPUT_DIR=submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2" \
  CLUSTER_PARTITION=hiperf \
  CLUSTER_NODE=tesla-a100 \
  CLUSTER_GPUS=1
```

Параметры кластера можно изменить через `CLUSTER_NODES`, `CLUSTER_CPUS`, `CLUSTER_MEMORY`, `CLUSTER_TIME`, `CLUSTER_JOB_NAME`, `CLUSTER_LOG_DIR`, `CLUSTER_GRES` и `CLUSTER_CUDA_DEVICES`. Значение `CLUSTER_GPUS` должно совпадать с числом GPU, доступным заданию. Если кластер использует другой тип ресурса GPU, передайте его через `CLUSTER_GRES`.

## Обучение

В ветке `original-geopose` используются авторские модель, функции потерь, конфигурации и полный `folds.csv` на 5923 примера из ARG, ATL, JAX и OMA. Источник: [решение автора, коммит 3ec4abc](https://github.com/drivendataorg/overhead-geopose-challenge/tree/3ec4abc256fc32c9e9d022aa5cc1478000a6aaae/1st%20Place). Для этого разбиения нужен соответствующий исходный тренировочный набор.

Для обучения используются скрипты `dist_train.sh` и `dist_train_tune.sh` с запуском через `torch.distributed.run`. `train.sh` объединяет подготовку TIFF-файлов, обучение и дообучение fold 5, затем повторяет оба этапа для fold 6:

```shell
make all
./train.sh /data/train /data/train_tiffs 4
```

Аргументы `train.sh`:

- `data_dir` — каталог с исходным тренировочным набором;
- `w_data_dir` — каталог для преобразованных TIFF-файлов;
- `num_gpus` — количество GPU для обучения.

В исходных данных ожидаются `CITY_ID_RGB.j2k`, `CITY_ID_AGL.tif` и `CITY_ID_VFLOW.json`. Подготовка переводит AGL из сантиметров в метры и сохраняет RGB как TIFF без уменьшения разрешения. Для исходного RGB в TIFF передайте `TRAIN_RGB_SUFFIX=tif`; уже подготовленные данные повторно не конвертируйте, а запускайте этапы через `dist_train.sh` и `dist_train_tune.sh`.

| Параметр | Первый этап | Дообучение |
| --- | --- | --- |
| Конфигурация | `configs/v2l.json` | `configs/v2l_tune.json` |
| Эпохи | 100 | 75 |
| Crop | 896 × 896 | 1792 × 1792 |
| Batch train / val на GPU | 4 / 2 | 2 / 2 |
| Learning rate | 0.0001 | 0.00003 |
| Multiplier датасета | 4 | 1 |

Первый этап автоматически загружает ImageNet-21k-веса энкодера `tf_efficientnetv2_l_in21k` через `timm`. Дообучение использует лучший checkpoint первого этапа своего fold. Опубликованные веса из TLDR для начала обучения не требуются. Результаты записываются в `weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2` и `..._6_r2`; обучение может перезаписать скачанные файлы с этими именами. Последние состояния сохраняются с суффиксом `_last`.

Скрипты автоматически используют `.venv/bin/python`; другой интерпретатор можно передать через `PYTHON`. Текстовые логи сохраняются в `logs/l5` и `logs/l6`, логи TensorBoard — в `logs/`; каталоги создаются автоматически. При ошибке обучение останавливается.

Для отправки обучения на Slurm-кластер из каталога проекта:

```shell
./run_run.sh /data/train /data/train_tiffs 1
```

Если конвертация уже успешно завершилась, используйте тот же каталог подготовленных данных и пропустите её:

```shell
SKIP_PREPARE=1 ./run_run.sh /data/train /data/train_tiffs 1
```

По умолчанию запрашиваются 1 GPU, 8 CPU и 40 ГБ оперативной памяти на узле `tesla-a100`, partition `hiperf`. Параметры можно изменить через `CLUSTER_PARTITION`, `CLUSTER_NODE`, `CLUSTER_CPUS`, `CLUSTER_MEMORY`, `CLUSTER_TIME`, `CLUSTER_GRES`; третий аргумент задаёт число GPU. Оценка использует число CPU из `SLURM_CPUS_PER_TASK`. Лимит 20 часов является настройкой задания, а не оценкой длительности обучения.

Автор использовал 4 × RTX A6000. Конфигурации сохраняют его batch size независимо от числа GPU: запуск на одной GPU уменьшает общий batch и может потребовать больше доступной видеопамяти, чем есть на выбранной карте. Для точного сравнения нужно учитывать также исходное число GPU и окружение; зависимости в этой ветке адаптированы для локального запуска.

## Проверка на тестовом наборе

Для предсказания используйте те же команды, что и для обычного запуска:

```shell
make predict \
  DATA_DIR=/data/test \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2" \
  GPUS=1
```

Параметры `GPUS` и `TILE_SIZE` позволяют адаптировать запуск к объёму доступной видеопамяти. Для обычных потребительских GPU может потребоваться уменьшить размер тайла.

## Очистка

Удалить собранные Cython-файлы и временные артефакты сборки:

```shell
make clean
```

Удалить виртуальное окружение:

```shell
make clean-venv
```
