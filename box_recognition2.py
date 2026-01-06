import os 
import cv2
from statistics import median
from ultralytics import YOLO

# ============================
# МОДЕЛЬ РАСПОЗНАВАНИЯ СИМВОЛОВ
# ============================
MODEL_PATH = os.getenv("REC_MODEL_PATH", "models/recognition.pt")
rec_model = YOLO(MODEL_PATH)


# ============================
# УТИЛИТЫ
# ============================
def _only_alnum_upper(s: str) -> str:
    return "".join(c for c in (s or "").upper() if c.isalnum())


def _detect_line_count_and_group(char_items, min_lines=2, max_lines=3):
    """
    Группировка символов в 2–3 строки (ТОЛЬКО для горизонтальных номеров)
    char_items: [{x1, y1, cx, cy, h, label}, ...]
    """
    if len(char_items) < 6:
        return None

    avg_h = sum(c["h"] for c in char_items) / len(char_items)
    if avg_h <= 1e-6:
        return None

    ys = sorted(c["cy"] for c in char_items)
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    big_gaps = [g for g in gaps if g > 0.6 * avg_h]
    n_lines = 1 + len(big_gaps)

    if not (min_lines <= n_lines <= max_lines):
        return None

    tol = 0.6 * avg_h
    items = sorted(char_items, key=lambda c: c["cy"])

    rows = []
    for ch in items:
        placed = False
        for row in rows:
            row_y = sum(x["cy"] for x in row) / len(row)
            if abs(ch["cy"] - row_y) <= tol:
                row.append(ch)
                placed = True
                break
        if not placed:
            rows.append([ch])

    if not (min_lines <= len(rows) <= max_lines):
        return None

    rows.sort(key=lambda row: sum(x["cy"] for x in row) / len(row))
    for row in rows:
        row.sort(key=lambda c: c["x1"])

    return rows


# ============================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================
def recognize_box(box, det_results, img):
    cls_idx = int(box.cls[0])
    cls_name = det_results.names[cls_idx]

    if cls_name not in ("container_number_h", "container_number_v"):
        return None

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    # ---------- OCR ----------
    rec_results = rec_model(
        roi,
        imgsz=640,
        conf=0.3,
        verbose=False
    )[0]

    char_items = []
    for cbox in rec_results.boxes:
        label = rec_results.names[int(cbox.cls[0])]
        cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0].tolist())
        h = max(1, cy2 - cy1)

        char_items.append({
            "x1": cx1,
            "y1": cy1,
            "cx": (cx1 + cx2) / 2,
            "cy": (cy1 + cy2) / 2,
            "h": h,
            "label": label
        })

    if len(char_items) < 6:
        return None

    # ---------- фильтрация мусора ----------
    med_h = median(c["h"] for c in char_items)
    char_items = [c for c in char_items if c["h"] >= 0.6 * med_h]

    if len(char_items) < 6:
        return None

    # =====================================================
    # ============ HORIZONTAL CONTAINER NUMBER ============
    # =====================================================
    if cls_name == "container_number_h":

        # 1) multi-line (2–3 строки)
        rows = _detect_line_count_and_group(char_items, min_lines=2, max_lines=3)
        if rows:
            recognized = "".join(ch["label"] for row in rows for ch in row)
            recognized = _only_alnum_upper(recognized)
            if len(recognized) >= 10:
                return recognized

        # 2) fallback: слева направо
        char_items.sort(key=lambda c: c["x1"])
        recognized = "".join(c["label"] for c in char_items)
        return _only_alnum_upper(recognized)

    # =====================================================
    # ============== VERTICAL CONTAINER NUMBER ============
    # =====================================================
    else:
        # 1) основной путь: сверху вниз
        char_items.sort(key=lambda c: c["y1"])
        recognized = "".join(c["label"] for c in char_items)
        recognized = _only_alnum_upper(recognized)

        # 2) fallback: поворот ROI
        if len(recognized) < 10:
            roi_rot = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            rec2 = rec_model(
                roi_rot,
                imgsz=640,
                conf=0.3,
                verbose=False
            )[0]

            chars2 = []
            for cbox in rec2.boxes:
                label = rec2.names[int(cbox.cls[0])]
                cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0].tolist())
                chars2.append((cy1, label))

            if len(chars2) >= 6:
                chars2.sort(key=lambda t: t[0])
                rotated = _only_alnum_upper("".join(ch for _, ch in chars2))
                if len(rotated) > len(recognized):
                    return rotated

        return recognized
