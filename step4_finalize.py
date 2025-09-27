# step4_finalize.py
"""
Step-4 Finalizer (robust version)

Place this file next to ocr_extract.py, ocr_test.py, and your Step-3 file(s).
Runs Step-1 -> Step-2 -> Step-3 (if available) and writes step4_final.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# import Step-1 and Step-2
try:
    from ocr_extract import extract_text_from_image_file_path
except Exception as e:
    raise RuntimeError("Cannot import extract_text_from_image_file_path from ocr_extract.py") from e

try:
    from ocr_test import normalize_tests
except Exception as e:
    raise RuntimeError("Cannot import normalize_tests from ocr_test.py") from e

# Try to import different Step-3 module/name permutations safely
generate_patient_summary_fn = None
tried = []
# Candidate module names and function names to try
module_candidates = ["ocr_summarize_groq", "ocr_summarize_langchain", "ocr_summarize", "ocr_summarize_groq"]
func_candidates = ["generate_patient_summary_groq", "generate_patient_summary_chain", "generate_patient_summary", "generate_patient_summary_gs", "generate_patient_summary"]

for modname in module_candidates:
    try:
        mod = __import__(modname)
    except Exception:
        continue
    for fname in func_candidates:
        if hasattr(mod, fname):
            generate_patient_summary_fn = getattr(mod, fname)
            break
    if generate_patient_summary_fn:
        break

# If still not found, leave None and we will fallback later
if generate_patient_summary_fn is None:
    print("⚠️ Step-3 summary function not found in common modules. The finalizer will use a safe local fallback summary if needed.")

# Local fallback summary builder (non-diagnostic)
def local_fallback_summary(normalized_tests):
    explanations = []
    notable = []
    for t in normalized_tests:
        name = t.get("name") or "Test"
        status = t.get("status")
        if status in ("low", "high"):
            notable.append(f"{status} {name}")
            if status == "low":
                explanations.append(f"{name} is below the typical range; possible non-diagnostic causes include diet or blood loss.")
            else:
                explanations.append(f"{name} is above the typical range; this can occur with infection or inflammation.")
    if not notable:
        return {"summary": "All reported test values appear within expected ranges.", "explanations": ["Values appear within expected ranges."], "confidence": 0.6}
    return {"summary": " and ".join(notable).capitalize() + ".", "explanations": explanations, "confidence": 0.7}

def build_final_output(normalized_json: Dict[str, Any],
                       summary_json: Optional[Dict[str, Any]],
                       ocr_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    final = {
        "tests": normalized_json.get("tests", []),
        "normalization_confidence": normalized_json.get("normalization_confidence", 0.0),
        "summary": None,
        "explanations": [],
        "summary_confidence": 0.0,
        "status": "ok",
        "ocr": ocr_meta or {}
    }

    if summary_json is None:
        final["status"] = "unprocessed"
        final["reason"] = "no_summary"
        return final

    if isinstance(summary_json, dict) and summary_json.get("status") == "unprocessed":
        final["status"] = "unprocessed"
        final["reason"] = summary_json.get("reason", "verification_failed")
        final["summary"] = None
        final["explanations"] = summary_json.get("explanations", [])
        final["summary_confidence"] = float(summary_json.get("confidence", 0.0)) if summary_json.get("confidence") else 0.0
        return final

    final["summary"] = summary_json.get("summary")
    final["explanations"] = summary_json.get("explanations", [])
    final["summary_confidence"] = float(summary_json.get("confidence", 0.0)) if summary_json.get("confidence") else 0.0

    final["_meta"] = {
        "timestamp": int(time.time()),
        "normalize_confidence": final["normalization_confidence"],
        "summary_confidence": final["summary_confidence"]
    }

    # Guardrail: low-confidence => unprocessed
    if final["normalization_confidence"] < 0.30 or final["summary_confidence"] < 0.30:
        final["status"] = "unprocessed"
        final["reason"] = "low_confidence"

    return final

def run_full_pipeline_and_finalize(image_path: str, force_confidence: Optional[float] = None) -> Dict[str, Any]:
    # Step-1 OCR
    ocr_out = extract_text_from_image_file_path(image_path, force_confidence=force_confidence)
    step1 = {
        "tests_raw": ocr_out.get("tests_raw", []),
        "confidence": ocr_out.get("confidence", 0.0),
        "engine_used": ocr_out.get("engine_used", "unknown")
    }

    # Step-2 Normalize
    norm = normalize_tests(step1["tests_raw"], step1["confidence"])

    # Step-3 Summarize: use available function if present, else fallback local
    if generate_patient_summary_fn is None:
        summary = local_fallback_summary(norm.get("tests", []))
        # mark that it's a fallback
        summary["note"] = "local_fallback_used"
    else:
        try:
            # call Step-3 function; many Step-3 functions accept (tests, tests_raw) or (normalized_tests, tests_raw)
            try:
                summary = generate_patient_summary_fn(norm.get("tests", []), step1["tests_raw"])
            except TypeError:
                # try alternate signature
                summary = generate_patient_summary_fn(norm, step1["tests_raw"])
        except Exception as e:
            print("Warning: Step-3 function raised exception:", e)
            summary = local_fallback_summary(norm.get("tests", []))
            summary["note"] = "local_fallback_due_to_error"

    # Build final
    final = build_final_output(norm, summary, ocr_meta=step1)

    # Save final
    try:
        with open("step4_final.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print("Saved final output to step4_final.json")
    except Exception as e:
        print("Warning saving file:", e)

    # Print to console
    print("\n=== Step-4 (Final Output) ===")
    print(json.dumps(final, indent=2, ensure_ascii=False))

    return final

# CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step4_finalize.py <image_path> [force_confidence]")
        sys.exit(1)
    image_path = sys.argv[1]
    force_conf = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_full_pipeline_and_finalize(image_path, force_confidence=force_conf)
