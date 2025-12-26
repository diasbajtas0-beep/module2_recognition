import cv2
from ultralytics import YOLO

MODEL_PATH = "models/recognition.pt"
rec_model = YOLO(MODEL_PATH)


def _only_alnum_upper(s: str) -> str:
    return "".join(c for c in (s or "").upper() if c.isalnum())


def _detect_line_count_and_group(char_items, min_lines=2, max_lines=3):
    """
    char_items: list of dicts with keys: cx, cy, h, label
    Возвращает rows (список строк, каждая строка — список char_items), если строк 2-3.
    Иначе None.
    """
    if len(char_items) < 6:
        return None

    ys = sorted([c["cy"] for c in char_items])
    avg_h = sum(c["h"] for c in char_items) / len(char_items)
    if avg_h <= 1e-6:
        return None

    # ищем большие разрывы по Y
    gaps = []
    for i in range(len(ys) - 1):
        gaps.append(ys[i + 1] - ys[i])

    big_gaps = [g for g in gaps if g > 0.6 * avg_h]
    n_lines = 1 + len(big_gaps)

    if not (min_lines <= n_lines <= max_lines):
        return None

    # группируем в строки жадно по близости Y
    tol = 0.6 * avg_h
    items = sorted(char_items, key=lambda d: d["cy"])

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

    # сортируем строки сверху вниз
    rows.sort(key=lambda row: sum(x["cy"] for x in row) / len(row))
    # сортируем внутри строки слева направо
    for row in rows:
        row.sort(key=lambda d: d["cx"])

    # после жадной группировки может получиться 4 строки из-за шума — тогда отбой
    if not (min_lines <= len(rows) <= max_lines):
        return None

    return rows


def recognize_box(box, det_results, img):
    cls_idx = int(box.cls[0])
    cls_name = det_results.names[cls_idx]

    if cls_name not in ("container_number_h", "container_number_v"):
        return None

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    rec_results = rec_model(
        roi,
        imgsz=640,
        conf=0.3,
        verbose=False
    )[0]

    # соберём расширенные данные по символам
    char_items = []
    char_boxes_base = []  # как раньше: (cx1, cy1, label)

    for cbox in rec_results.boxes:
        c_cls_idx = int(cbox.cls[0])
        label = rec_results.names[c_cls_idx]

        cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0].tolist())
        w = max(1, cx2 - cx1)
        h = max(1, cy2 - cy1)
        cx = (cx1 + cx2) / 2.0
        cy = (cy1 + cy2) / 2.0

        char_boxes_base.append((cx1, cy1, label))
        char_items.append({"cx": cx, "cy": cy, "h": h, "label": label})

    if not char_boxes_base:
        return None

    # -------------------------
    # 1) БАЗОВОЕ (как у тебя было)
    # -------------------------
    if cls_name == "container_number_h":
        char_boxes_base.sort(key=lambda t: t[0])  # по X
    else:
        char_boxes_base.sort(key=lambda t: t[1])  # по Y

    recognized_base = "".join(ch for _, _, ch in char_boxes_base)
    recognized_base = _only_alnum_upper(recognized_base)

    # -------------------------
    # 2) ДОП. АВТОКОРРЕКЦИЯ: если видно 2-3 строки
    # -------------------------
    rows = _detect_line_count_and_group(char_items, min_lines=2, max_lines=3)
    if rows is not None:
        recognized_multi = "".join(ch["label"] for row in rows for ch in row)
        recognized_multi = _only_alnum_upper(recognized_multi)

        # выбираем что вернуть:
        # обычно multi даёт более правильный порядок и длину
        # если вдруг multi короче/пустой — оставим базовый
        if recognized_multi and len(recognized_multi) >= len(recognized_base):
            return recognized_multi

    return recognized_base
