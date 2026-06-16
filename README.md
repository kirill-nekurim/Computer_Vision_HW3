# Computer Vision HW3 + HW3.5

## Отчёт

**PDF:** [`notebooks/hw3_server.pdf`](notebooks/hw3_server.pdf)

> GitHub **не показывает PDF в браузере** (ошибка preview — это нормально).  
> Откройте файл через **Download raw file** или клонируйте репозиторий.

Полный отчёт также можно смотреть в этом README (краткие результаты ниже).

## HW3 — DETR на COCO-subset (10 классов)

Модель: `facebook/detr-resnet-50`, fine-tuning 20 эпох.

| Метрика | Значение |
|---------|----------|
| mAP (best epoch 20) | 0.341 |
| mAP@50 | 0.488 |
| mAP@75 | 0.351 |

Классы: person, bicycle, car, bus, truck, cat, dog, bottle, cup, chair.

## HW3.5 — Stable Diffusion + ControlNet

Редкие классы: **truck**, **bicycle**, **chair**.  
Сгенерировано по 30 синтетических изображений на класс (SD 1.5 + ControlNet Canny).  
Классификатор: ResNet-18, ablation baseline vs baseline + synthetic.

| Эксперимент | Accuracy | Macro-F1 | Mean recall (rare) |
|-------------|----------|----------|--------------------|
| Baseline | 85.9% | 0.667 | 0.746 |
| + Synthetic | 84.7% | 0.632 | 0.714 |
| Δ | −1.3 pp | −0.035 | −0.032 |

## Вывод

DETR успешно дообучен на подмножестве COCO.  
Синтетические данные **не дали прироста** — вероятная причина domain gap между SD-генерациями и реальными COCO crops.

## Артефакты

- `src/` — код обучения и генерации
- `outputs/metrics/` — метрики DETR
- `outputs/hw35/ablation.json` — ablation HW3.5
- `outputs/hw35/synthetic_examples.png` — примеры синтетики
- `data/synthetic/` — синтетические изображения
