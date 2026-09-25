# Geopose Challenge — решение, занявшее первое место

## Обзор

Решение в основном основано на [статье CVPR EarthVision 2021](https://arxiv.org/abs/2105.08229) и [статье CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf).

## Требования

Проект запускается локально в виртуальном окружении Python. По умолчанию используется Python 3.9, а для вычислений на GPU необходимы совместимые драйвер NVIDIA и CUDA.

Зависимости из `req.txt` закреплены по локальному окружению проекта. Установка `GDAL` и `pygdal` также требует системных библиотек GDAL. Makefile по умолчанию использует индекс пакетов PyTorch для CUDA 12.4; при необходимости его можно изменить через `PIP_EXTRA_INDEX_URL`.

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

Веса не входят в репозиторий, поэтому передайте пути к ним явно:

```shell
make predict \
  DATA_DIR=/data/images \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="/data/weights/model-a.pt /data/weights/model-b.pt" \
  RGB_SUFFIX=png \
  GPUS=1
```

По умолчанию `make predict` также принимает `BATCH_SIZE`, `DOWNSAMPLE`, `UNIT`, `TILE_SIZE`, `COMPRESS` и `MASTER_PORT`. Например, `TILE_SIZE=1024` уменьшает потребление памяти GPU, а `COMPRESS=false` отключает постобработку и упаковку.

Те же параметры можно передать непосредственно в `predict.sh`:

```shell
./predict.sh \
  --weights /data/weights/model-a.pt /data/weights/model-b.pt \
  --input-dir /data/images \
  --output-dir /data/submission \
  --rgb-suffix png \
  --gpus 1 \
  --tile-size 2048
```

Если используются локальные веса с именами из репозитория, можно указать их путём:

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

## Обучение

Для обучения используются скрипты `dist_train.sh` и `dist_train_tune.sh`. `train.sh` объединяет подготовку TIFF-файлов и два запуска обучения:

```shell
./train.sh /data/train /data/train_tiffs 4
```

Аргументы `train.sh`:

- `data_dir` — каталог с исходным тренировочным набором;
- `w_data_dir` — каталог для преобразованных TIFF-файлов;
- `num_gpus` — количество GPU для обучения.

Первый запуск использует конфигурацию `configs/v2l.json` и размер crop `896x896`, второй — `configs/v2l_tune.json` и размер crop `1792x1792`. Количество эпох и параметры оптимизатора задаются в этих конфигурациях. Текущий сценарий `train.sh` запускает fold 1; запуск второго fold отключён.

Текстовые логи и логи TensorBoard сохраняются в каталоге `logs/`. Перед запуском обучения убедитесь, что этот каталог существует.

## Проверка на тестовом наборе

Для предсказания используйте те же команды, что и для обычного запуска:

```shell
make predict \
  DATA_DIR=/data/test \
  OUTPUT_DIR=/data/submission \
  WEIGHTS="/data/weights/model-a.pt /data/weights/model-b.pt" \
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
