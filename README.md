# Medical Report Simplifier (OCR → Normalize → Summarize → Finalize)

## Overview
This project ingests scanned or typed medical lab reports, extracts test lines using OCR,
normalizes test names/units/ranges, verifies results with an LLM, generates patient-friendly
explanations, and produces a final JSON output suitable for downstream systems.

Steps:
1. **Step-1 (OCR)** — `ocr_extract.py`
   - Preprocessing (resize, denoise, threshold, deskew)
   - Ocr engines: Tesseract preferred, EasyOCR fallback
   - Outputs `tests_raw` (list of cleaned lines) and `confidence` (0..1)

2. **Step-2 (Normalization)** — `ocr_test.py`
   - Regex-based parser to extract name/value/unit/status
   - Canonical lookup and reference ranges
   - Returns `tests` list and `normalization_confidence`

3. **Step-3 (Summarize)** — `ocr_summarize_groq.py` (or `ocr_summarize_langchain.py`)
   - Verify LLM chain to ensure normalized tests are grounded in OCR
   - Summarize tests in patient-friendly language
   - Conservative fallback if LLM is unavailable or verification fails

4. **Step-4 (Finalize)** — `step4_finalize.py`
   - Combine normalized tests + summary into final JSON
   - Add `_meta` provenance and `status` guardrails

## Quick start (Windows)
1. Create and activate venv:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
