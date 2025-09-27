# ocr_summarize_groq.py
"""
Step-3 pipeline (separate file).
Runs:
  1) ocr_extract.extract_text_from_image_file_path
  2) ocr_test.normalize_tests
  3) Groq LLM verify -> summarize (via groq_client.llm)
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

# Step-1 and Step-2 imports (must be in same folder)
from ocr_extract import extract_text_from_image_file_path
from ocr_test import normalize_tests

# Groq client import (separate file)
try:
    from groq_client import llm, ChatPromptTemplate, JsonOutputParser
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"⚠️ groq_client import failed: {e}")
    GROQ_AVAILABLE = False
    llm = None
    ChatPromptTemplate = None
    JsonOutputParser = None

# Prompt templates
VERIFY_SYSTEM = (
    "You are a strict JSON-only verifier. Return only a single JSON object. "
    "Check whether the normalized tests (name + value) exist in the raw OCR lines and that no extra tests were invented."
)
VERIFY_USER_TEMPLATE = (
    "Raw OCR lines:\n{tests_raw}\n\nNormalized tests (name + value):\n{normalized_brief}\n\n"
    "If OK return exactly: {\"ok\": true}\n"
    "Else return exactly: {\"ok\": false, \"reason\": \"<brief reason>\"}\n"
    "Return JSON only."
)

SUMMARIZE_SYSTEM = (
    "You are a medical-language assistant. Produce ONLY valid JSON. "
    "Do NOT give diagnosis or treatment — only patient-friendly plain-language explanations."
)
SUMMARIZE_USER_TEMPLATE = (
    "Normalized tests JSON:\n{normalized_full}\n\n"
    "Return JSON with keys: 'summary' (one short sentence), 'explanations' (array of 1-4 short sentences), 'confidence' (0..1).\n"
    "Do NOT invent tests. Return JSON only."
)


def _brief_normalized(tests: List[Dict[str, Any]]) -> str:
    lines = []
    for t in tests:
        name = t.get("name") or ""
        value = t.get("value")
        unit = t.get("unit") or ""
        value_str = "" if value is None else str(value)
        lines.append(" ".join([p for p in (name, value_str, unit) if p]).strip())
    return "\n".join(lines)


def _full_normalized_json(tests: List[Dict[str, Any]]) -> str:
    return json.dumps(tests, ensure_ascii=False)


def _extract_first_json(text: str) -> Optional[dict]:
    """
    Try to extract the first JSON object from `text`.
    Returns dict on success, None on failure.
    """
    if not text or "{" not in text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    candidate = text[s:e + 1]
    try:
        return json.loads(candidate)
    except Exception:
        # try replacing single quotes with double quotes (last resort)
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None


def _llm_invoke_text(prompt_text: str) -> str:
    """
    Safely call the `llm` object. Handles two common shapes:
      - llm.generate(prompt_text) -> object (with .text or .generations)
      - llm(prompt_text) -> string
    Returns the raw text output (string).
    """
    if not GROQ_AVAILABLE or llm is None:
        raise RuntimeError("Groq client not available")

    # If llm has 'generate', call it and try to extract text
    try:
        if hasattr(llm, "generate"):
            raw = llm.generate(prompt_text)
            # Try some common attributes
            if isinstance(raw, str):
                return raw
            # many SDKs have .text or .outputs or .generations
            if hasattr(raw, "text"):
                return getattr(raw, "text")
            if hasattr(raw, "output_text"):
                return getattr(raw, "output_text")
            if hasattr(raw, "generations"):
                # try to join generation texts
                gens = getattr(raw, "generations")
                if isinstance(gens, list) and gens:
                    first = gens[0]
                    # generation element may be dict-like or have .text
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
                    if hasattr(first, "text"):
                        return getattr(first, "text")
            # fallback to str()
            return str(raw)
        # else if llm is callable
        elif callable(llm):
            out = llm(prompt_text)
            return out if isinstance(out, str) else str(out)
        else:
            return str(llm)
    except Exception as e:
        # propagate with context
        raise RuntimeError(f"LLM invocation failed: {e}")


# Groq verifier and summarizer (LangChain-style attempt then fallback)
def _run_groq_verifier(tests_raw: List[str], normalized_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client not available")
    # try chain style (if ChatPromptTemplate and JsonOutputParser are available)
    try:
        if ChatPromptTemplate and JsonOutputParser:
            prompt = ChatPromptTemplate.from_messages([
                ("system", VERIFY_SYSTEM),
                ("human", VERIFY_USER_TEMPLATE.format(
                    tests_raw="\n".join(tests_raw),
                    normalized_brief=_brief_normalized(normalized_tests)
                ))
            ])
            try:
                chain = prompt | llm | JsonOutputParser()
                parsed = chain.invoke({})
                return parsed
            except Exception:
                # fallback to raw text prompt
                pass

        # fallback: call llm directly
        text_prompt = VERIFY_SYSTEM + "\n\n" + VERIFY_USER_TEMPLATE.format(
            tests_raw="\n".join(tests_raw),
            normalized_brief=_brief_normalized(normalized_tests)
        )
        raw_text = _llm_invoke_text(text_prompt)
        parsed = _extract_first_json(raw_text)
        if parsed is None:
            raise RuntimeError("Could not parse verifier JSON from Groq output.")
        return parsed
    except Exception as e:
        raise RuntimeError(f"Groq verifier failed: {e}")


def _run_groq_summarizer(normalized_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client not available")
    try:
        if ChatPromptTemplate and JsonOutputParser:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SUMMARIZE_SYSTEM),
                ("human", SUMMARIZE_USER_TEMPLATE.format(normalized_full=_full_normalized_json(normalized_tests)))
            ])
            try:
                chain = prompt | llm | JsonOutputParser()
                parsed = chain.invoke({})
                return parsed
            except Exception:
                # fallback to raw prompt
                pass

        text_prompt = SUMMARIZE_SYSTEM + "\n\n" + SUMMARIZE_USER_TEMPLATE.format(
            normalized_full=_full_normalized_json(normalized_tests)
        )
        raw_text = _llm_invoke_text(text_prompt)
        parsed = _extract_first_json(raw_text)
        if parsed is None:
            raise RuntimeError("Could not parse summary JSON from Groq output.")
        return parsed
    except Exception as e:
        raise RuntimeError(f"Groq summarizer failed: {e}")


def _local_fallback_summary(normalized_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    explanations = []
    notable = []
    for t in normalized_tests:
        name = t.get("name") or "Test"
        status = (t.get("status") or "").lower()
        if status in ("low", "high"):
            notable.append(f"{status} {name}")
            if status == "low":
                explanations.append(f"{name} is below the typical range; possible non-diagnostic causes include diet or blood loss.")
            else:
                explanations.append(f"{name} is above the typical range; this can occur with infection or inflammation.")
    if not notable:
        return {
            "summary": "All reported test values appear within expected ranges.",
            "explanations": ["Values appear within expected ranges."],
            "confidence": 0.6
        }
    return {
        "summary": " and ".join(notable).capitalize() + ".",
        "explanations": explanations,
        "confidence": 0.7
    }


def generate_patient_summary_groq(normalized_tests: List[Dict[str, Any]], tests_raw: List[str]) -> Dict[str, Any]:
    # verify
    try:
        if GROQ_AVAILABLE:
            verifier = _run_groq_verifier(tests_raw, normalized_tests)
            if not verifier.get("ok", False):
                return {
                    "status": "unprocessed",
                    "reason": verifier.get("reason", "verification_failed"),
                    "verification": verifier
                }
        else:
            # local check: ensure the first 3 chars of name appear in OCR (lowercase)
            joined = " ".join([r.lower() for r in tests_raw])
            for t in normalized_tests:
                name = (t.get("name") or "").lower()
                if name and name[:3] not in joined:
                    return {
                        "status": "unprocessed",
                        "reason": f"{t.get('name')} not found in OCR (local check)"
                    }
    except Exception as e:
        print(f"Verifier error: {e} — falling back to local verification.")

    # summarize
    if GROQ_AVAILABLE:
        try:
            return _run_groq_summarizer(normalized_tests)
        except Exception as e:
            print(f"Groq summarizer error: {e} — falling back to local summary.")
            return _local_fallback_summary(normalized_tests)
    else:
        return _local_fallback_summary(normalized_tests)


# CLI runner: run steps 1 -> 2 -> 3
def run_all(image_path: str, force_conf: Optional[float] = None):
    ocr_out = extract_text_from_image_file_path(image_path, force_confidence=force_conf)
    step1 = {
        "tests_raw": ocr_out.get("tests_raw", []),
        "confidence": ocr_out.get("confidence"),
        "engine_used": ocr_out.get("engine_used")
    }
    print("\n=== Step-1 (OCR) ===")
    print(json.dumps(step1, indent=2, ensure_ascii=False))

    norm = normalize_tests(step1["tests_raw"], step1["confidence"])
    print("\n=== Step-2 (Normalized) ===")
    print(json.dumps(norm, indent=2, ensure_ascii=False))

    if not norm.get("tests"):
        print("\nNo normalized tests found; skipping summarization.")
        return {"ocr": step1, "normalize": norm, "summary": {"status": "unprocessed", "reason": "no_tests"}}

    summary = generate_patient_summary_groq(norm["tests"], step1["tests_raw"])
    print("\n=== Step-3 (Patient-friendly summary) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    try:
        with open("step3_groq_result.json", "w", encoding="utf-8") as f:
            json.dump({"ocr": step1, "normalize": norm, "summary": summary}, f, indent=2, ensure_ascii=False)
        print("\nSaved combined result to step3_groq_result.json")
    except Exception:
        pass

    return {"ocr": step1, "normalize": norm, "summary": summary}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_summarize_groq.py <image_path> [force_confidence]")
        sys.exit(1)
    path = sys.argv[1]
    fc = float(sys.argv[2]) if len(sys.argv) > 2 else None
    run_all(path, force_conf=fc)
