# streamlit_app.py
"""
Medical Report Simplifier — Streamlit UI (simplified sidebar)
Placed next to:
  - ocr_extract.py
  - ocr_test.py
  - ocr_summarize_groq.py (or ocr_summarize_langchain.py)
  - step4_finalize.py  (optional)

How I run it locally:
  venv\Scripts\activate
  pip install -r requirements.txt
  streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime

# ---- Optional dotenv support (I keep keys locally in .env while developing) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Page config ----
st.set_page_config(
    page_title="Medical Report Simplifier — Ajay",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- App header ----
header_col, help_col = st.columns([0.9, 0.1])
with header_col:
    st.title("Medical Report Simplifier")
    st.markdown("**Quick pipeline:** OCR → normalize tests → LLM summary → final JSON")
with help_col:
    st.caption(f"Local run • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# ---------------- Try to import pipeline functions ----------------
extract_fn = None
normalize_fn = None
summary_fn = None
finalize_fn = None

# Step-1: OCR
try:
    from ocr_extract import extract_text_from_image_file_path as extract_text_from_image_file_path
    extract_fn = extract_text_from_image_file_path
except Exception:
    st.sidebar.error("ocr_extract.py not available (OCR disabled).")

# Step-2: Normalization
try:
    from ocr_test import normalize_tests as normalize_tests
    normalize_fn = normalize_tests
except Exception:
    st.sidebar.error("ocr_test.py not available (Normalization disabled).")

# Step-3: Summary (try multiple candidate modules/function names)
summary_candidates = [
    ("ocr_summarize_groq", "generate_patient_summary_groq"),
    ("ocr_summarize_langchain", "generate_patient_summary_chain"),
    ("ocr_summarize_langchain", "generate_patient_summary"),
    ("ocr_summarize", "generate_patient_summary"),
    ("ocr_summarize_gs", "generate_patient_summary_gs"),
]
for modname, fname in summary_candidates:
    try:
        mod = __import__(modname)
        if hasattr(mod, fname):
            summary_fn = getattr(mod, fname)
            st.sidebar.success(f"Step-3: Using {modname}.{fname}")
            break
    except Exception:
        pass

if summary_fn is None:
    st.sidebar.warning("No Step-3 LLM function found — I will use a conservative local fallback for summaries.")

# Step-4: Finalize (optional)
try:
    from step4_finalize import build_final_output as build_final_output
    finalize_fn = build_final_output
except Exception:
    finalize_fn = None

# ---------------- Sidebar (simplified as requested) ----------------
st.sidebar.header("Settings")
# Only this control is visible in the sidebar per request:
max_upload_size_mb = st.sidebar.number_input("Max upload size (MB)", min_value=1, max_value=100, value=12)

# Add the run/submit button to the sidebar as the "2nd attached" control
run_btn_sidebar = st.sidebar.button("Submit")

# ---------------- Internal defaults for removed controls ----------------
# (kept so the rest of the UI code works without the removed sidebar controls)
use_llm = True          # previously controlled by a checkbox
force_conf = 0.0        # previously a slider
show_raw = True         # previously a checkbox
ocr_engine = "auto"     # previously a selectbox

# ---------------- Utility helpers ----------------
def local_summary_fallback(normalized_tests):
    explanations = []
    notable = []
    for t in normalized_tests:
        name = t.get("name") or t.get("test_name") or "Test"
        status = t.get("status")
        if status in ("low", "high"):
            notable.append(f"{status} {name}")
            if status == "low":
                explanations.append(f"{name} is below the typical range; non-diagnostic causes include diet, recent bleeding, or lab variability.")
            else:
                explanations.append(f"{name} is above the typical range; common causes include infection, inflammation, or lab variability.")
    if not notable:
        return {"summary": "All reported test values appear within expected ranges.", "explanations": ["Values appear within expected ranges."], "confidence": 0.6}
    return {"summary": " and ".join(notable).capitalize() + ".", "explanations": explanations, "confidence": 0.72}

def _run_ocr(filepath, force_confidence=None):
    if extract_fn:
        try:
            try:
                return extract_fn(filepath, engine=ocr_engine, force_confidence=force_confidence)
            except TypeError:
                return extract_fn(filepath, force_confidence)
        except Exception as e:
            st.warning(f"OCR function raised an error: {e}")
            return {"tests_raw": [], "confidence": 0.0, "engine_used": "error"}
    else:
        return {"tests_raw": [], "confidence": float(force_confidence or 0.0), "engine_used": "none"}

def _normalize(tests_raw, confidence):
    if normalize_fn:
        try:
            return normalize_fn(tests_raw, confidence)
        except Exception as e:
            st.warning(f"Normalization error: {e}")
            return {"tests": [], "normalization_confidence": 0.0}
    return {"tests": [], "normalization_confidence": 0.0}

def _summarize(norm, tests_raw):
    if use_llm and summary_fn:
        try:
            try:
                return summary_fn(norm.get("tests", []), tests_raw)
            except TypeError:
                return summary_fn(norm, tests_raw)
        except Exception as e:
            st.warning(f"LLM summary raised error: {e}. Falling back to local summary.")
            out = local_summary_fallback(norm.get("tests", []))
            out["note"] = "fallback_due_to_llm_error"
            return out
    else:
        if use_llm and not summary_fn:
            st.warning("LLM requested but summary function not found — using local fallback.")
        return local_summary_fallback(norm.get("tests", []))

def _finalize(norm, summary, ocr_meta):
    if finalize_fn:
        try:
            return finalize_fn(norm, summary, ocr_meta=ocr_meta)
        except TypeError:
            pass
    return {
        "tests": norm.get("tests", []),
        "normalization_confidence": norm.get("normalization_confidence", 0.0),
        "summary": summary.get("summary"),
        "explanations": summary.get("explanations", []),
        "summary_confidence": summary.get("confidence", 0.0),
        "ocr_meta": ocr_meta,
        "status": "ok"
    }

# ---------------- File upload area (main page) ----------------
col1, col2 = st.columns([0.75, 0.25])
with col1:
    uploaded_file = st.file_uploader("Upload medical report image (png/jpg/jpeg/tif/tiff)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    # Keep a local Run button as well so user has two ways to run (sidebar button or main page)
    run_btn_main = st.button("Run Pipeline")
with col2:
    st.caption("Upload the image and click Submit (sidebar) or Run (this page).")

st.divider()

# ---------------- Run pipeline logic ----------------
def run_pipeline_on_file(filepath, force_confidence=None):
    progress = st.progress(0, text="Starting pipeline...")
    step = 0

    # Step 1: OCR
    step += 1
    progress.progress(step / 4, text="Running OCR (Step 1)...")
    st.info("Step 1 — OCR")
    ocr_out = _run_ocr(filepath, force_confidence)
    step1 = {
        "tests_raw": ocr_out.get("tests_raw", []),
        "confidence": float(ocr_out.get("confidence", 0.0)),
        "engine_used": ocr_out.get("engine_used", "unknown")
    }
    st.metric(label="OCR engine", value=step1["engine_used"])
    st.metric(label="OCR confidence", value=f"{step1['confidence']:.2f}")
    if show_raw:
        with st.expander("Raw OCR output — expand to inspect"):
            st.json(step1)

    # Step 2: Normalize
    step += 1
    progress.progress(step / 4, text="Normalizing tests (Step 2)...")
    st.info("Step 2 — Normalization")
    norm = _normalize(step1["tests_raw"], step1["confidence"])
    with st.expander("Normalized tests"):
        st.json(norm)

    # Step 3: Summarize
    step += 1
    progress.progress(step / 4, text="Generating patient-friendly summary (Step 3)...")
    st.info("Step 3 — Summary")
    summary = _summarize(norm, step1["tests_raw"])
    with st.expander("Patient-friendly summary"):
        st.json(summary)

    # Step 4: Finalize
    step += 1
    progress.progress(step / 4, text="Building final output (Step 4)...")
    st.info("Step 4 — Final JSON")
    final = _finalize(norm, summary, ocr_meta=step1)
    with st.expander("Final output JSON"):
        st.json(final)

    progress.empty()
    st.success("Pipeline finished.")
    return final

# Button handling: runs if either the main or sidebar button is clicked
if run_btn_main or run_btn_sidebar:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        uploaded_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if uploaded_size_mb > max_upload_size_mb:
            st.error(f"Uploaded file is {uploaded_size_mb:.1f} MB which exceeds max {max_upload_size_mb} MB.")
        else:
            suffix = Path(uploaded_file.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                t.write(uploaded_file.getbuffer())
                temp_path = t.name

            try:
                final_result = run_pipeline_on_file(temp_path, force_confidence=(force_conf if force_conf > 0 else None))

                filename = f"medical_report_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button(
                    "Download final JSON",
                    data=json.dumps(final_result, ensure_ascii=False, indent=2),
                    file_name=filename,
                    mime="application/json"
                )

            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
