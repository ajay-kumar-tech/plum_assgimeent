import cv2, numpy as np, os, json
from PIL import Image
import pytesseract
from pytesseract import Output
import easyocr

# ========== CONFIG ==========
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PREFERRED_ENGINE = "tesseract"      # "tesseract" or "easyocr"
TESSERACT_MIN_CONF = 0.45
# If tesseract installed at default path, set for pytesseract
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
# ============================

def preprocess_image_bytes(img_bytes, target_height=1200):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < target_height:
        scale = target_height / float(h)
        gray = cv2.resize(gray, (int(w*scale), target_height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    coords = np.column_stack(np.where(th > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h2, w2) = th.shape
        center = (w2 // 2, h2 // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        th = cv2.warpAffine(th, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return th

def ocr_with_tesseract_bytes(img_bytes, lang='eng'):
    img_proc = preprocess_image_bytes(img_bytes)
    pil = Image.fromarray(img_proc)
    data = pytesseract.image_to_data(pil, lang=lang, output_type=Output.DICT)
    n = len(data['text'])
    lines_map = {}
    for i in range(n):
        text = data['text'][i].strip()
        conf_raw = data['conf'][i]
        try:
            conf = float(conf_raw) if conf_raw.strip() != "" and conf_raw != "-1" else -1.0
        except:
            conf = -1.0
        if text == "":
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        lines_map.setdefault(key, {"words": [], "confs": []})
        lines_map[key]["words"].append(text)
        lines_map[key]["confs"].append(conf)
    lines = []
    for key in sorted(lines_map.keys()):
        words = lines_map[key]["words"]
        confs = lines_map[key]["confs"]
        text_line = " ".join(words)
        valid = [c for c in confs if c >= 0]
        avg_conf = (sum(valid)/len(valid)/100.0) if valid else 0.0
        lines.append({"text": text_line, "confidence": round(avg_conf, 2)})
    agg_conf = round(sum([l["confidence"] for l in lines]) / max(1, len(lines)), 2) if lines else 0.0
    return {"engine": "tesseract", "lines": lines, "aggregate_confidence": agg_conf}

def ocr_with_easyocr_bytes(img_bytes, langs=['en']):
    reader = easyocr.Reader(langs, gpu=False)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = reader.readtext(img)
    lines = []
    for bbox, text, conf in results:
        lines.append({"text": text.strip(), "confidence": round(float(conf), 2)})
    agg_conf = round(sum([l["confidence"] for l in lines]) / max(1, len(lines)), 2) if lines else 0.0
    return {"engine": "easyocr", "lines": lines, "aggregate_confidence": agg_conf}

def _clean_raw_line(s: str) -> str:
    """Fix minor typos and formatting in raw OCR line for downstream parsing."""
    if not s:
        return s
    # remove common OCR comma in numbers (e.g., 11,200 -> 11200)
    s = s.replace(",", "")
    # fix double spaces
    s = " ".join(s.split())
    # common misspellings (expand as needed)
    s = s.replace("Hemglobin", "Hemoglobin").replace("Hemogloin", "Hemoglobin")
    # ensure a space before unit if combined like '10.2g/dL'
    s = s.replace("g/dL(", " g/dL(").replace("g/dL", "g/dL")
    return s.strip()

def extract_text_from_image_file_path(path: str, force_confidence: float = None):
    """Main function:
       - runs OCR (tesseract preferred, easyocr fallback)
       - cleans lines (remove thousands commas etc.)
       - returns dict matching your Step-1 expected JSON
       force_confidence: optional float 0..1 to force the 'confidence' value for demo
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        img_bytes = f.read()

    if PREFERRED_ENGINE == "tesseract":
        try:
            res = ocr_with_tesseract_bytes(img_bytes)
            if res["aggregate_confidence"] < TESSERACT_MIN_CONF:
                res2 = ocr_with_easyocr_bytes(img_bytes)
                chosen = res2 if res2["aggregate_confidence"] > res["aggregate_confidence"] else res
            else:
                chosen = res
        except Exception:
            chosen = ocr_with_easyocr_bytes(img_bytes)
    else:
        try:
            chosen = ocr_with_easyocr_bytes(img_bytes)
            if chosen["aggregate_confidence"] < 0.4:
                chosen = ocr_with_tesseract_bytes(img_bytes)
        except Exception:
            chosen = ocr_with_tesseract_bytes(img_bytes)

    # Clean the lines for better downstream parsing
    lines = chosen.get("lines", [])
    cleaned = []
    for l in lines:
        txt = _clean_raw_line(l.get("text", ""))
        conf = l.get("confidence", 0.0)
        cleaned.append({"text": txt, "confidence": conf})

    tests_raw = [c["text"] for c in cleaned]
    agg_conf = float(chosen.get("aggregate_confidence", 0.0))
    if force_confidence is not None:
        agg_conf = float(force_confidence)
    agg_conf = round(agg_conf, 2)

    out = {
        "tests_raw": tests_raw,
        "confidence": agg_conf,
        "engine_used": chosen.get("engine", "unknown"),
        "lines": cleaned
    }
    return out

# If run directly you can test:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_extract.py <image_path> [force_confidence]")
        sys.exit(1)
    path = sys.argv[1]
    fc = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(extract_text_from_image_file_path(path, force_confidence=fc), indent=2, ensure_ascii=False))
