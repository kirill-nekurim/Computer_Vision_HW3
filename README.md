# HW3: Fine-tuning DETR на COCO-subset

Fine-tuning `facebook/detr-resnet-50` на подмножестве COCO 2017 (10 классов) с TensorBoard, profiler trace, метриками mAP/mAP50 и error analysis.

## Классы

`person`, `bicycle`, `car`, `bus`, `truck`, `cat`, `dog`, `bottle`, `cup`, `chair`

COCO ID: bottle=**44**, cup=**47**, chair=**62** (не 39/41/56).

## Структура

```
src/
  config.py              # гиперпараметры и пути
  prepare_coco_subset.py # фильтрация COCO → 10 классов
  dataset.py             # PyTorch Dataset для DETR
  train.py               # обучение + TensorBoard + profiler
  evaluate.py            # mAP / mAP50
  error_analysis.py      # ошибки классификации и локализации
  visualize.py           # отрисовка боксов
scripts/
  download_coco.py       # скачивание COCO 2017
notebooks/
  hw3_detr_colab.ipynb   # полный пайплайн для Colab GPU
```

## Локально (Mac / CPU)

### 1. Окружение

```bash
cd hw3-detr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Скачать только аннотации (~250 MB) — этого достаточно для subset

```bash
python scripts/download_coco.py --annotations-only
```

Полный COCO (~20 GB) **не обязателен**. Для домашки нужен subset из 10 классов.

### 3. Подготовить subset (например, 25% данных)

```bash
# Только JSON-аннотации, без картинок
python -m src.prepare_coco_subset \
  --skip-image-copy \
  --train-fraction 0.05 \
  --val-fraction 0.05

# Скачать только нужные картинки (~1-3 GB вместо 20 GB)
python scripts/download_subset_images.py
```

Можно задать лимит явно:

```bash
python -m src.prepare_coco_subset \
  --skip-image-copy \
  --max-train-images 2000 \
  --max-val-images 500
python scripts/download_subset_images.py
```

Если уже скачан полный COCO локально:

```bash
python -m src.prepare_coco_subset --coco-dir data/coco --output-dir data/coco_subset
```

### 4. Проверить импорты (без обучения)

```bash
python -c "from src.train import build_model; print('OK')"
```

> **Обучение на CPU очень медленное.** Для fine-tuning используйте Colab-ноутбук.

## Google Colab (рекомендуется)

1. Загрузите проект на GitHub **или** zip-архивом в Colab.
2. Откройте `notebooks/hw3_detr_colab.ipynb`.
3. Runtime → **T4 GPU**.
4. В первой ячейке укажите `GITHUB_REPO` или загрузите проект в `/content/hw3-detr`.
5. Запустите все ячейки по порядку.
6. Скачайте `hw3_detr_artifacts.zip` и добавьте `outputs/` в репозиторий.

## Запуск отдельных этапов (на GPU-машине)

```bash
# Обучение
python -m src.train --epochs 20 --batch-size 4

# Метрики
python -m src.evaluate --checkpoint outputs/checkpoints/final

# Error analysis
python -m src.error_analysis --checkpoint outputs/checkpoints/final

# TensorBoard
tensorboard --logdir outputs/tensorboard
```

Profiler trace: `outputs/profiler/trace.json` → открыть в Chrome (`chrome://tracing`).

## Гиперпараметры по умолчанию

| Параметр | Значение |
|----------|----------|
| Модель | `facebook/detr-resnet-50` |
| Epochs | 20 |
| Batch size | 4 |
| LR (head) | 1e-5 |
| LR (backbone) | 1e-5 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Score threshold | 0.5 |
| IoU threshold (error analysis) | 0.5 |

## Что сдавать

- [x] `src/` — код обучения и анализа
- [x] `README.md` — этот файл + ваши наблюдения после эксперимента
- [ ] `outputs/tensorboard/` — логи TensorBoard
- [ ] `outputs/profiler/trace.json` — trace профайлера
- [ ] `outputs/metrics/metrics_val.json` — mAP, mAP50
- [ ] `outputs/plots/losses.png` — графики loss
- [ ] `outputs/visualizations/` — примеры ошибок

## Шаблон таблицы метрик (заполнить после Colab)

| Model | Epochs | mAP | mAP50 |
|-------|--------|-----|-------|
| DETR-resnet-50 (fine-tuned) | 20 | — | — |

## Наблюдения (заполнить после эксперимента)

- Classification loss (`loss_ce`) ...
- Bbox losses (`loss_bbox`, `loss_giou`) ...
- Типичные classification errors: ...
- Типичные localization errors: ...

---

# HW3.5: Синтетические данные (Stable Diffusion + ControlNet)

Аугментация редких классов **truck**, **bicycle**, **chair** синтетикой и ablation на ResNet-18.

> **Исправление маппинга классов:** в ранней версии `config.py` были неверные COCO ID
> (`bottle=39`, `cup=41`, `chair=56` — baseball bat / skateboard / broccoli).
> Исправлено на `bottle=44`, `cup=47`, `chair=62`. Subset и crops пересобраны.

## Ноутбук для защиты

`notebooks/hw35_synthetic_server.ipynb` — короткий ноутбук для показа HW3.5 на сдаче:
распределение crops, pipeline генерации, галерея синтетики, ablation-таблица и график.

## Структура HW3.5

```
src/
  hw35_config.py           # параметры SD + классификатора
  prepare_cls_crops.py     # crop объектов из COCO для классификации
  cls_dataset.py           # Dataset (real / real+synthetic)
  train_classifier.py      # ablation baseline vs synthetic
scripts/
  generate_synthetic.py    # Stable Diffusion + ControlNet (Canny)
  run_hw35.sh              # полный пайплайн
notebooks/
  hw35_synthetic_server.ipynb  # защита HW3.5
data/
  cls_crops/               # реальные crops (train/val по классам)
  synthetic/               # синтетика для truck, bicycle, chair
outputs/hw35/
  baseline/                # метрики без синтетики
  with_synthetic/          # метрики с синтетикой
  ablation.json            # таблица сравнения
  synthetic_examples.png   # галерея примеров
```

## Запуск

```bash
source .venv/bin/activate

# 1. Crops из COCO (после prepare_coco_subset)
python -m src.prepare_cls_crops

# 2. Генерация синтетики (нужен GPU, ~30–60 мин на 30 img/class)
python scripts/generate_synthetic.py --num-images 30 --classes truck bicycle chair

# 3. Ablation: baseline + with synthetic
python -m src.train_classifier --epochs 15

# Или всё сразу:
bash scripts/run_hw35.sh
```

## Модели

| Компонент | Значение |
|-----------|----------|
| SD | `runwayml/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/sd-controlnet-canny` |
| Классификатор | ResNet-18 (ImageNet init) |
| Synthetic classes | truck, bicycle, chair |

## Ablation (V100, 30 synthetic images/class, ResNet-18, 15 epochs)

| Experiment | Train samples | Accuracy | Macro-F1 | Recall truck | Recall bicycle | Recall chair |
|------------|---------------|----------|----------|--------------|----------------|--------------|
| Baseline | 6492 | 0.919 | 0.733 | 0.622 | 0.765 | 1.000 |
| + Synthetic | 6582 (+90) | 0.928 | 0.776 | 0.649 | 0.882 | 1.000 |

**Delta (+ synthetic − baseline):** accuracy +0.89 pp, macro-F1 +4.4 pp, mean recall (rare) +4.8 pp.

## Выводы

- Синтетика через SD + ControlNet (Canny) **улучшила** recall для `truck` (+2.7 pp) и `bicycle` (+11.8 pp).
- `chair` уже был на 100% recall в baseline — синтетика не дала прироста, но и не ухудшила.
- Общая accuracy и macro-F1 выросли; синтетика помогает редким классам без деградации на val (только реальные данные).
- После исправления COCO ID (`chair=62`, `bottle=44`, `cup=47`) crops содержат реальные объекты, а не брокколи/skateboard.
