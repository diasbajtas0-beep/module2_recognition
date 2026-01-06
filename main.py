import os
import json
from collections import deque
from statistics import median
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

from box_recognition2 import recognize_box
from tools import validate_code


# =========================================
# КОНФИГУРАЦИЯ - ВЫБОР ИСТОЧНИКА И РЕЖИМА
# =========================================

class Config:
    

    # РЕЖИМ ИСТОЧНИКА
    SOURCE_TYPE = "video"  # "video" или "camera"

    # Если SOURCE_TYPE = "video"
    VIDEO_PATH = os.getenv("VIDEO_PATH", "video-test.mp4")

    # Если SOURCE_TYPE = "camera"
    CAMERA_ID = 0  # 0 для USB камеры, или IP адрес/RTSP для IP камеры

    # РЕЖИМ ОБРАБОТКИ ВРЕМЕНИ
    TIME_MODE = "live"  # "video" (текущее время) или "live" (время с потока)

    # ВЫХОДНЫЕ ФАЙЛЫ
    JSON_OUT = os.getenv("JSON_OUT", "result.json")
    OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", "output.mp4")

    # ПАРАМЕТРЫ ОБРАБОТКИ
    MIN_STABLE_FRAMES = 8      # валидных кадров для фиксации контейнера
    MAX_GAP_FRAMES = 10        # подряд кадров можно "потерять"
    GAP_MULT = 1.8             # множитель для детекции пропущенных контейнеров
    PLACEHOLDER = ""           # значение для пропущенных контейнеров

    # МОДЕЛЬ
    MODEL_PATH = os.getenv("DET_MODEL_PATH", "models/detection.pt")

    # ПАРАМЕТРЫ ДЕТЕКЦИИ
    DETECTION_CONF = 0.1       # confidence порог
    DETECTION_IOU = 0.75       # NMS IOU порог
    DETECTION_IMGSZ = 1280     # размер изображения
    DETECTION_DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" или "cpu"

    # ЛОГИРОВАНИЕ
    VERBOSE = True             # выводить ли информацию в консоль


def validate_config():
    """Проверка корректности конфигурации"""
    if Config.SOURCE_TYPE not in ["video", "camera"]:
        raise ValueError(f"Invalid SOURCE_TYPE: {Config.SOURCE_TYPE}")

    if Config.SOURCE_TYPE == "video" and not os.path.exists(Config.VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {Config.VIDEO_PATH}")

    if Config.TIME_MODE not in ["video", "live"]:
        raise ValueError(f"Invalid TIME_MODE: {Config.TIME_MODE}")


# ===========================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ВРЕМЕНЕМ
# ===========================

def get_container_timestamp(frame_idx: int, fps: float, start_time=None):
    """
    Получить временную метку контейнера.
    TIME_MODE:
      - "video": текущее системное время
      - "live": start_time + frame_idx/fps
    """
    if Config.TIME_MODE == "video":
        return datetime.now().isoformat()

    # live
    if start_time is None:
        start_time = datetime.now()

    elapsed = timedelta(seconds=frame_idx / fps) if fps else timedelta(seconds=0)
    return (start_time + elapsed).isoformat()


def create_container_entry(container_code: str, frame_idx: int, fps: float, start_time=None):
    """
    Создать запись контейнера для JSON.
    ВАЖНО: frame_number НЕ добавляем (как ты просил).
    """
    return {
        "container_number": container_code,
        "detected_at": get_container_timestamp(frame_idx, fps, start_time),
    }


def log(message: str):
    """Выводить сообщение если VERBOSE=True"""
    if Config.VERBOSE:
        print(message)


# =============================================
# ФУНКЦИЯ ОБРАБОТКИ ОДНОГО КАДРА
# =============================================

def _process_frame(
    img,
    r,
    frame_idx,
    fps,
    start_time,
    out,
    state,
    final_containers,
    gap_history,
):
    """
    Обработать один кадр видео/потока.

    state: dict с ключами current_code, good_run, gap_run, last_final, last_final_frame
    """
    current_code = state["current_code"]
    good_run = state["good_run"]
    gap_run = state["gap_run"]
    last_final = state["last_final"]
    last_final_frame = state["last_final_frame"]

    # 1) Поиск лучшего кода на текущем кадре
    best_code = None
    best_conf = -1.0
    best_bbox = None

    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = map(int, r.boxes[i].xyxy[0].tolist())
        det_conf = float(r.boxes.conf[i])

        detected = recognize_box(r.boxes[i], r, img)
        if detected is None:
            continue

        code = detected.strip().upper()
        if not validate_code(code):
            continue

        if det_conf > best_conf:
            best_conf = det_conf
            best_code = code
            best_bbox = (x1, y1, x2, y2)

    # 2) Обновление состояния
    if best_code is None:
        if current_code is not None:
            gap_run += 1
            if gap_run > Config.MAX_GAP_FRAMES:
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

    # 3) Фиксируем контейнер
    if good_run == Config.MIN_STABLE_FRAMES:
        if current_code != last_final:
            if last_final_frame is not None:
                delta = frame_idx - last_final_frame

                if len(gap_history) >= 5:
                    med = median(gap_history)
                    if med > 0 and delta > med * Config.GAP_MULT:
                        missed = max(1, int(round(delta / med)) - 1)
                        for _ in range(missed):
                            placeholder_entry = create_container_entry(
                                Config.PLACEHOLDER, frame_idx, fps, start_time
                            )
                            final_containers.append(placeholder_entry)
                            log(f"[MISS] placeholder at frame {frame_idx}")

                if delta > 0:
                    gap_history.append(delta)

            container_entry = create_container_entry(
                current_code, frame_idx, fps, start_time
            )
            final_containers.append(container_entry)
            last_final = current_code
            last_final_frame = frame_idx

            log(f"[CONTAINER] #{len(final_containers)} {current_code} @ frame {frame_idx}")

    # 4) Отрисовка и запись
    if best_code is not None and best_bbox is not None:
        x1, y1, x2, y2 = best_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{best_code} {good_run}/{Config.MIN_STABLE_FRAMES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        f"{Config.TIME_MODE.upper()} | Frame: {frame_idx}",
        (10, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    out.write(img)

    # Обновляем state
    state["current_code"] = current_code
    state["good_run"] = good_run
    state["gap_run"] = gap_run
    state["last_final"] = last_final
    state["last_final_frame"] = last_final_frame
    return state


# ====================================
# ИНИЦИАЛИЗАЦИЯ ИСТОЧНИКА И ПАРАМЕТРОВ
# ====================================

validate_config()

model = YOLO(Config.MODEL_PATH)

# Узнаем fps/размер
if Config.SOURCE_TYPE == "video":
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
else:
    cap = cv2.VideoCapture(Config.CAMERA_ID)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fallback если камера/видео не отдали размеры
if width <= 0 or height <= 0:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame to detect resolution")
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

log(f"\n{'=' * 60}")
log(f"SOURCE: {Config.SOURCE_TYPE.upper()}")
log(f"TIME_MODE: {Config.TIME_MODE.upper()}")
log(f"FPS: {fps}, Resolution: {width}x{height}")
log(f"{'=' * 60}\n")

cap.release()

# VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(Config.OUTPUT_VIDEO, fourcc, int(fps), (width, height))

# Генератор результатов
if Config.SOURCE_TYPE == "video":
    results = model.predict(
        source=Config.VIDEO_PATH,
        conf=Config.DETECTION_CONF,
        iou=Config.DETECTION_IOU,
        max_det=100,
        imgsz=Config.DETECTION_IMGSZ,
        stream=True,
        device=Config.DETECTION_DEVICE,
        verbose=False,
    )
else:
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    results = None

start_time = datetime.now() if Config.TIME_MODE == "live" else None

state = {
    "current_code": None,
    "good_run": 0,
    "gap_run": 0,
    "last_final": None,
    "last_final_frame": None,
}

final_containers = []
gap_history = deque(maxlen=20)
frame_idx = 0

try:
    if Config.SOURCE_TYPE == "video":
        for frame_idx, r in enumerate(results):
            img = r.orig_img.copy()
            state = _process_frame(
                img, r, frame_idx, fps, start_time, out,
                state, final_containers, gap_history
            )
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                log("[INFO] Конец потока камеры или ошибка чтения")
                break

            pred = model([frame])[0]
            pred.orig_img = frame

            state = _process_frame(
                frame, pred, frame_idx, fps, start_time, out,
                state, final_containers, gap_history
            )

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == 27:
                log("[INFO] Пользователь прервал обработку")
                break

        cap.release()
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    log("\n[INTERRUPTED] Обработка прервана пользователем")

finally:
    out.release()

    
    json_output = {
        "mode": Config.TIME_MODE,
        "source": Config.SOURCE_TYPE,
        "total_containers": len(final_containers),
        "start_time": start_time.isoformat() if start_time else None,
        "containers": [
            {
                "sequence_number": idx + 1,  # <-- с 1
                "container_number": entry["container_number"],
                "detected_at": entry["detected_at"],
            }
            for idx, entry in enumerate(final_containers)
        ],
    }

    with open(Config.JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    log(f"\n{'=' * 60}")
    log(f"[OK] Video saved: {Config.OUTPUT_VIDEO}")
    log(f"[OK] JSON saved: {Config.JSON_OUT}")
    log(f"[OK] Total containers: {len(final_containers)}")
    log(f"{'=' * 60}\n")

    if final_containers:
        log("First 5 containers:")
        for i, entry in enumerate(final_containers[:5], 1):
            log(f"  {i}. {entry['container_number']:12} @ {entry['detected_at']}")
