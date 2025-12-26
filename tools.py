# tools.py

LETTER_VALUES = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38,
}

def iso6346_check_digit(code10: str) -> int:
    """
    code10: первые 10 символов кода контейнера (без контрольной цифры)
    возвращает ожидаемую контрольную цифру 0..9, или -1 если встретили невалидный символ
    """
    s = 0
    for i, ch in enumerate(code10):
        if ch.isdigit():
            v = int(ch)
        else:
            v = LETTER_VALUES.get(ch, None)
            if v is None:
                return -1
        s += v * (2 ** i)
    return (s % 11) % 10

def validate_code(code: str) -> bool:
    if not isinstance(code, str):
        return False

    code = code.strip().upper()
    if len(code) != 11:
        return False

    owner = code[:3]
    if not owner.isalpha():
        return False

    category = code[3]
    if not category.isalpha():
        return False

    serial = code[4:10]
    if not serial.isdigit():
        return False

    check_digit = code[10]
    if not check_digit.isdigit():
        return False

    expected = iso6346_check_digit(code[:10])
    if expected < 0:
        return False 

    return int(check_digit) == expected
