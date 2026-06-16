# Запуск на сервере с GPU (V100)

## Рекомендация: Jupyter-ноутбук

**Для защиты лучше использовать:**
- `notebooks/hw3_detr_server.ipynb` — HW3 (DETR)
- `notebooks/hw35_synthetic_server.ipynb` — HW3.5 (SD + ControlNet, ablation)

В ноутбуке DETR наглядно:
- распределение классов и проверка bbox;
- dashboard (loss + mAP по эпохам);
- gallery предсказаний и error analysis;
- чеклист артефактов для сдачи.

Скрипты (`run_server.sh`, `run_overnight.sh`) удобны для **ночного прогона в tmux**, когда ноутбук не нужен.

```bash
cd /root/hw3-detr
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

Kernel: `.venv/bin/python`. Откройте ноутбук и выберите режим `MODE = "full"` (20 эпох).

---

## 1. С Mac — отправить проект на сервер

```bash
cd ~
zip -r hw3-detr.zip hw3-detr \
  -x "hw3-detr/.venv/*" \
  -x "hw3-detr/data/*" \
  -x "hw3-detr/outputs/*" \
  -x "hw3-detr/**/__pycache__/*"

scp hw3-detr.zip root@vm5:/root/
```

## 2. На сервере — распаковать

```bash
ssh vm5
cd /root
unzip -q -o hw3-detr.zip
cd hw3-detr
```

## 3. Проверить GPU

```bash
nvidia-smi
```

## 4. Запустить в tmux (ночной прогон без ноутбука)

```bash
tmux new -s detr
chmod +x scripts/run_overnight.sh
./scripts/run_overnight.sh
```

Отсоединиться: `Ctrl+B`, затем `D`  
Вернуться: `tmux attach -t detr`

После обучения откройте ноутбук и запустите ячейки **5–10** (визуализация и защита).

## Параметры (опционально)

```bash
EPOCHS=20 BATCH_SIZE=4 MAX_TRAIN=2500 MAX_VAL=400 LR_BACKBONE=1e-6 ./scripts/run_server.sh
```

### Ночной прогон (по умолчанию)

- **2500 train + 400 val**
- **20 эпох**
- `lr=1e-5`, `lr_backbone=1e-6`
- best checkpoint по **mAP@50** → `outputs/checkpoints/best`

## 5. Скачать результаты на Mac

```bash
scp -r root@vm5:/root/hw3-detr/outputs ~/hw3-detr/
```

## Важно: перезапуск после исправления bbox

Если обучали **до исправления** `dataset.py`, удалите старые чекпоинты:

```bash
rm -rf outputs/checkpoints
```

И переобучите с нуля.

## Если torch не видит CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## По шагам (CLI)

```bash
source .venv/bin/activate

python scripts/download_coco.py --annotations-only
python -m src.prepare_coco_subset --skip-image-copy --max-train-images 2500 --max-val-images 400
python scripts/download_subset_images.py
python -m src.train --epochs 20 --batch-size 4 --lr 1e-5 --lr-backbone 1e-6
python -m src.evaluate --checkpoint outputs/checkpoints/best
python -m src.error_analysis --checkpoint outputs/checkpoints/best
```
