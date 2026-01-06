# container_code_recognition_video

Детекция и распознавание номера контейнера (YOLO + recognition).

## Input
Видео для теста:
https://drive.google.com/file/d/1bNnlDTv_HpOAozhx-Ti7VWYF5p-w-Ksb/view?usp=sharing

## Output
Пример результатов (видео + txt/json):
https://drive.google.com/drive/folders/1GMNUI0KtN4irSZwqg8wiLG6xY3EHRrWJ?usp=sharing

---

## Структура проекта
- `main.py` — основной запуск пайплайна
- `box_recognition2.py` — распознавание символов в ROI
- `tools.py` — валидация/утилиты
- `models/` — веса моделей (detection/recognition)
- `data/` — вход/выход (не коммитится)

---

## Docker

### Build
```bash
docker build -f Dockerfile.gpu -t container-ocr:gpu .


---


### Run (GPU), Требуется установленный NVIDIA Driver + NVIDIA Container Toolkit.

mkdir -p data
# положи входное видео
cp /path/to/video.mp4 data/input.mp4

docker run --rm --gpus all \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/data" \
  -e VIDEO_PATH=/data/input.mp4 \
  -e JSON_OUT=/data/result.json \
  -e OUTPUT_VIDEO=/data/output.mp4 \
  -e DEVICE=cuda \
  container-ocr:gpu

--- 