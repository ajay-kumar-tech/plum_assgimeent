# ocr_test.py
"""
Step-2: Normalization of tests
Input: Step-1 OCR output (tests_raw, confidence)
Output: Normalized JSON
"""

import re, json
from typing import List, Dict, Optional, Tuple
from difflib import get_close_matches

# Reference ranges (extend as needed)
REFERENCE_RANGES = {
    "Hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 15.0, "synonyms": ["hb", "hemog", "hemoglobin"]},
    "WBC": {"unit": "/uL", "low": 4000, "high": 11000, "synonyms": ["wbc", "white blood cell", "wbcs"]},
    "Platelets": {"unit": "/uL", "low": 150000, "high": 450000, "synonyms": ["plt", "platelets"]},
}
CANONICAL = list(REFERENCE_RANGES.keys())

# Regex for test line detection
TEST_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z\s\-\']{0,60}?)\s+"      
    r"([+-]?\d{1,3}(?:[,]\d{3})*(?:\.\d+)?)\s*"
    r"([a-zA-Z/µ%]{1,6})?\s*"
    r"(?:\(\s*([A-Za-z0-9\s\-]+)\s*\))?",
    flags=re.IGNORECASE
)

def sanitize(s: str) -> str:
    if not s: return ""
    return " ".join(s.replace("\xa0"," ").split()).strip()

def canonical_name_guess(name: str) -> Tuple[Optional[str], float]:
    if not name: return None, 0.0
    ng = name.strip().lower()
    for canon, info in REFERENCE_RANGES.items():
        if ng in [canon.lower()]+[s.lower() for s in info.get("synonyms", [])]:
            return canon, 1.0
    match = get_close_matches(name.title(), CANONICAL, n=1, cutoff=0.6)
    return (match[0], 0.8) if match else (None, 0.0)

def _record_from_match(m, src_idx: int) -> Dict:
    name_raw, val_raw, unit_raw, status_raw = m.groups()
    name_raw = sanitize(name_raw)

    # value
    val = None
    if val_raw:
        try:
            v = val_raw.replace(",","")
            val = float(v) if "." in v else int(v)
        except: pass

    # status
    status = None
    if status_raw:
        s = status_raw.lower()
        if "low" in s: status="low"
        elif "high" in s: status="high"
        elif "normal" in s: status="normal"

    # canonical name
    canon, _ = canonical_name_guess(name_raw)
    ref = REFERENCE_RANGES.get(canon)
    ref_range = {"low": ref["low"], "high": ref["high"]} if ref else None

    # infer status if missing
    if not status and val and ref:
        status = "low" if val < ref["low"] else "high" if val > ref["high"] else "normal"

    unit_out = unit_raw or (ref["unit"] if ref else None)
    if isinstance(val,float) and val.is_integer(): val=int(val)

    return {
        "name": canon.title() if canon else name_raw.title(),
        "value": val,
        "unit": unit_out,
        "status": status or "unknown",
        "ref_range": ref_range,
        "source_raw_index": src_idx
    }

def normalize_tests(tests_raw: List[str], ocr_conf: float) -> Dict:
    combined = "  ".join([sanitize(t) for t in tests_raw])
    matches = list(TEST_PATTERN.finditer(combined))
    parsed = [_record_from_match(m, 0) for m in matches]

    norm_conf = round(0.5*ocr_conf + 0.3*(len([p for p in parsed if p["value"]]) / max(1,len(parsed))) + 0.2,2)
    return {"tests": parsed, "normalization_confidence": norm_conf}

if __name__ == "__main__":
    # Demo usage
    demo = {"tests_raw": ["Hemoglobin 10.2 g/dL (Low)", "WBC 11200 /uL (High)"], "confidence":0.8}
    norm = normalize_tests(demo["tests_raw"], demo["confidence"])
    print(json.dumps(norm, indent=2, ensure_ascii=False))
