import os
from collections import deque
from statistics import median

import cv2
from ultralytics import YOLO

from box_recognition import recognize_box
from tools import validate_code

# download video
url = 'https://drive.google.com/file/d/1bNnlDTv_HpOAozhx-Ti7VWYF5p-w-Ksb/view?usp=sharing'
VIDEO_PATH = "/home/resolving/git_ml/container_code_recognition_video/video-test.mp4"    
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

TXT_OUT = "recognized_containers2.txt"
OUTPUT_VIDEO = "output_annotated.mp4"

MIN_STABLE_FRAMES = 8    # сколько валидных кадров нужно для фиксации контейнера
MAX_GAP_FRAMES = 10      # сколько подряд кадров можно "потерять", не сбрасывая контейнер

gap_history = deque(maxlen=20)   # история "нормальных" разрывов между контейнерами (в кадрах)
last_final_frame = None
GAP_MULT = 1.8                   # если разрыв > median_gap * GAP_MULT -> считаем что пропустили контейнер(ы)
PLACEHOLDER = ""                 # оставляем пусто; можно сделать "MISSING"

MODEL_PATH = "models/detection.pt"
model = YOLO(MODEL_PATH)

# Инициализируем VideoWriter
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

results = model.predict(
    source=VIDEO_PATH,
    conf=0.1,
    iou=0.75,
    max_det=100,
    imgsz=1280,
    stream=True,
    device="cuda",
    verbose=False
)

current_code = None
good_run = 0          # сколько валидных кадров накопили по current_code
gap_run = 0           # сколько подряд кадров current_code "не подтвердился"

final_codes = []
last_final = None

for frame_idx, r in enumerate(results):
    img = r.orig_img.copy()

    # 1) лучший код на кадре
    best_code = None
    best_conf = -1.0
    best_bbox = None

    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = map(int, r.boxes[i].xyxy[0].tolist())
        det_conf = float(r.boxes.conf[i])

        detected = recognize_box(r.boxes[i], r, img)
        if detected is None:
            continue

        code = detected.upper()
        if not validate_code(code):
            continue

        if det_conf > best_conf:
            best_conf = det_conf
            best_code = code
            best_bbox = (x1, y1, x2, y2)

    # 2) обновление состояния с допуском пропусков
    if best_code is None:
        if current_code is not None:
            gap_run += 1
            if gap_run > MAX_GAP_FRAMES:
                current_code = None
                good_run = 0
                gap_run = 0
    else:
        if current_code is None:
            current_code = best_code
            good_run = 1
            gap_run = 0
        elif best_code == current_code:
            good_run += 1
            gap_run = 0
        else:
            current_code = best_code
            good_run = 1
            gap_run = 0

        # 3) фиксируем контейнер
        if good_run == MIN_STABLE_FRAMES:
            if current_code != last_final:

                # --- вставка "пропусков" по большому разрыву ---
                if last_final_frame is not None:
                    delta = frame_idx - last_final_frame

                    # когда есть статистика, можем оценить "ненормальный" gap
                    if len(gap_history) >= 5:
                        med = median(gap_history)
                        if med > 0 and delta > med * GAP_MULT:
                            # сколько контейнеров могли пропустить (примерно)
                            missed = max(1, int(round(delta / med)) - 1)
                            for _ in range(missed):
                                final_codes.append(PLACEHOLDER)
                                print(f"[MISS] inserted placeholder before frame {frame_idx}")

                    # обновляем историю разрывов
                    if delta > 0:
                        gap_history.append(delta)

                # --- записываем контейнер ---
                final_codes.append(current_code)
                last_final = current_code
                last_final_frame = frame_idx
                print(f"[CONTAINER] #{len(final_codes)} {current_code} at frame {frame_idx}")

    # 4) отрисовка (контроль) и запись в видео
    if best_code is not None and best_bbox is not None:
        x1, y1, x2, y2 = best_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{best_code} good={good_run}/{MIN_STABLE_FRAMES} gap={gap_run}/{MAX_GAP_FRAMES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
    else:
        if current_code is not None:
            cv2.putText(
                img,
                f"{current_code} good={good_run}/{MIN_STABLE_FRAMES} gap={gap_run}/{MAX_GAP_FRAMES}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

    # Записываем кадр в видео
    out.write(img)

# Закрываем VideoWriter
out.release()

# Сохраняем результаты в текстовый файл
with open(TXT_OUT, "w", encoding="utf-8") as f:
    for idx, code in enumerate(final_codes, start=1):
        f.write(f"{idx}\t{code}\n")

print(f"[OK] saved video: {OUTPUT_VIDEO}")
print(f"[OK] saved txt: {TXT_OUT} ({len(final_codes)} containers)")
