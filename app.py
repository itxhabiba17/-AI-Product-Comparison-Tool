"""
AI Product Comparison Tool
--------------------------
Upload product specification files (PDF, TXT, DOCX), enter your requirements,
and let an AI (Groq LLM) compare the products using ONLY the uploaded documents.

Run with:  streamlit run app.py
"""

import io
import json
import os

import pandas as pd
import streamlit as st
from docx import Document          # reads .docx files
from groq import Groq              # Groq API client
from pypdf import PdfReader        # reads .pdf files

# =============================================================================
# 1. SETTINGS
# =============================================================================

# ---------------------------------------------------------------------------
# >>> GROQ API KEY - ADD YOUR KEY HERE (choose ONE of the three options) <<<
#
# Option A (recommended): create the file  .streamlit/secrets.toml  containing:
#       GROQ_API_KEY = "your-key-here"
#
# Option B: set an environment variable before running the app:
#       Windows (PowerShell):  $env:GROQ_API_KEY="your-key-here"
#       Mac / Linux:           export GROQ_API_KEY="your-key-here"
#
# Option C (quick testing only): paste it below instead of the placeholder.
#       Never share or upload this file if you do this!
# ---------------------------------------------------------------------------
GROQ_API_KEY_PLACEHOLDER = "PASTE_YOUR_GROQ_API_KEY_HERE"

MODEL_NAME = "llama-3.3-70b-versatile"   # a strong, free-tier-friendly Groq model
MAX_CHARS_PER_DOC = 6000                 # limit text per document (keeps requests small)
SUPPORTED_TYPES = ["pdf", "txt", "docx"]


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_api_key():
    """Find the Groq API key: Streamlit secrets -> environment variable -> placeholder."""
    # Option A: Streamlit secrets
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass  # no secrets file exists - that's fine

    # Option B: environment variable
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key

    # Option C: value pasted in this file (ignored if still the placeholder)
    if GROQ_API_KEY_PLACEHOLDER and GROQ_API_KEY_PLACEHOLDER != "PASTE_YOUR_GROQ_API_KEY_HERE":
        return GROQ_API_KEY_PLACEHOLDER

    return ""


def extract_text_from_pdf(file_bytes):
    """Read all pages of a PDF and return the text."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_txt(file_bytes):
    """Decode a plain text file (tries UTF-8 first, then Latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


def extract_text_from_docx(file_bytes):
    """Read paragraphs and tables from a Word document."""
    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:                       # specs are often stored in tables
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text(uploaded_file):
    """
    Pick the right reader for the file type.
    Returns (text, error_message). If everything is fine, error_message is None.
    """
    name = uploaded_file.name
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    file_bytes = uploaded_file.getvalue()

    if extension not in SUPPORTED_TYPES:
        return "", f"'{name}' is not supported. Please upload PDF, TXT, or DOCX files."

    try:
        if extension == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif extension == "txt":
            text = extract_text_from_txt(file_bytes)
        else:
            text = extract_text_from_docx(file_bytes)
    except Exception as error:
        return "", f"Could not read '{name}': {error}"

    if not text.strip():
        return "", (f"'{name}' appears to be empty (or it is a scanned PDF "
                    f"with no selectable text).")

    return text.strip(), None


def build_prompt(documents, requirements):
    """Create the instructions we send to the AI."""
    docs_text = ""
    for i, (filename, text) in enumerate(documents, start=1):
        docs_text += f"\n=== DOCUMENT {i}: {filename} ===\n{text}\n"

    return f"""
You are a careful product comparison assistant.

STRICT RULES:
- Use ONLY the information in the documents below. Do not use outside knowledge.
- NEVER invent or guess specifications. If something is missing, write exactly "Not available in document".
- Each document describes one product. Use the product name found in the document (or the file name if no name is found).
- Write in simple, easy-to-understand English.
- Judge advantages and disadvantages in relation to the USER REQUIREMENTS.

USER REQUIREMENTS:
{requirements}

DOCUMENTS:
{docs_text}

Return ONLY a valid JSON object (no extra text, no markdown) in this exact format:
{{
  "comparison_rows": [
    {{
      "Product": "product name",
      "Feature": "feature name (e.g. Price, Battery Life, Storage, Performance)",
      "Specification": "value from the document or 'Not available in document'",
      "Advantages": "short advantage for the user's needs, or '-'",
      "Disadvantages": "short disadvantage for the user's needs, or '-'"
    }}
  ],
  "similarities": ["short sentence", "..."],
  "differences": ["short sentence", "..."],
  "trade_offs": [
    {{
      "product": "product name",
      "you_gain": "what the user gains by choosing it",
      "you_lose": "what the user gives up by choosing it"
    }}
  ],
  "recommendation": {{
    "best_product": "product name",
    "reason": "clear explanation of why it best matches the requirements",
    "missing_information": ["important details not found in the documents"]
  }}
}}

Include a row for EVERY product for each important feature (use the same feature names across products so they can be compared).
"""


def call_groq(api_key, prompt):
    """Send the prompt to Groq and return the parsed JSON result."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",
             "content": "You compare products strictly using provided documents and reply only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,                              # low = more factual, less creative
        response_format={"type": "json_object"},      # ask Groq for JSON output
    )
    raw_text = response.choices[0].message.content

    # Safety: strip markdown fences if the model added them
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def show_list(items):
    """Show a list of strings as bullet points."""
    if not items:
        st.write("No information provided.")
        return
    for item in items:
        st.markdown(f"- {item}")


# =============================================================================
# 3. PAGE SETUP & STYLE
# =============================================================================

st.set_page_config(page_title="AI Product Comparison Tool", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
        .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
        .subtitle {font-size: 1.05rem; opacity: 0.75; margin-bottom: 1.2rem;}
        .rec-box {
            padding: 1.2rem 1.4rem; border-radius: 12px;
            border-left: 6px solid #2e7d32; background: rgba(46,125,50,0.10);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🛒 AI Product Comparison Tool</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload product specification files, tell us what you need, '
    "and get an AI comparison and recommendation based strictly on your documents.</p>",
    unsafe_allow_html=True,
)

# --- Sidebar: API key status + help -----------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    if api_key:
        st.success("Groq API key found ✅")
    else:
        st.warning("No Groq API key found.")
        api_key = st.text_input(
            "Enter your Groq API key (temporary)", type="password",
            help="Better: add GROQ_API_KEY to .streamlit/secrets.toml or an environment variable.",
        )
    st.caption(f"Model: `{MODEL_NAME}`")
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Upload 2 or more product files\n"
        "2. Write your requirements\n"
        "3. Click **Compare Products**\n"
        "4. Review, edit, and download the table"
    )

# =============================================================================
# 4. INPUT SECTION
# =============================================================================

left, right = st.columns(2, gap="large")

with left:
    st.subheader("📄 1. Upload Product Specifications")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files (one file per product)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

with right:
    st.subheader("📝 2. Your Requirements")
    requirements = st.text_area(
        "Describe what you need",
        height=170,
        placeholder="Example: Budget under $800, strong performance for video editing, "
                    "battery life of at least 10 hours, 512 GB storage or more.",
    )

compare_clicked = st.button("🔍 Compare Products", type="primary", use_container_width=True)

# =============================================================================
# 5. RUN THE COMPARISON
# =============================================================================

if compare_clicked:
    # --- Validate inputs -----------------------------------------------------
    if not api_key:
        st.error("Missing Groq API key. Add it to secrets, an environment variable, or the sidebar.")
    elif not uploaded_files or len(uploaded_files) < 2:
        st.error("Please upload at least 2 product specification files to compare.")
    elif not requirements.strip():
        st.error("Please enter your requirements.")
    else:
        # --- Extract text from every file -----------------------------------
        documents = []
        problems = []
        with st.spinner("Reading your documents..."):
            for file in uploaded_files:
                text, error = extract_text(file)
                if error:
                    problems.append(error)
                else:
                    if len(text) > MAX_CHARS_PER_DOC:
                        st.info(f"'{file.name}' is long, so only the first "
                                f"{MAX_CHARS_PER_DOC:,} characters were used.")
                        text = text[:MAX_CHARS_PER_DOC]
                    documents.append((file.name, text))

        for problem in problems:
            st.warning(problem)

        if len(documents) < 2:
            st.error("At least 2 readable documents are needed. Please fix the problems above.")
        else:
            # --- Ask the AI ---------------------------------------------------
            try:
                with st.spinner("AI is comparing the products..."):
                    prompt = build_prompt(documents, requirements)
                    st.session_state["result"] = call_groq(api_key, prompt)
            except json.JSONDecodeError:
                st.error("The AI returned an unreadable answer. Please click Compare again.")
            except Exception as error:
                st.error(f"Something went wrong while contacting Groq: {error}")

# =============================================================================
# 6. SHOW RESULTS (kept in session_state so edits don't erase them)
# =============================================================================

result = st.session_state.get("result")

if result:
    st.markdown("---")

    # --- Comparison table (editable) ------------------------------------------
    st.subheader("📊 Comparison Table")
    st.caption("You can click any cell to edit it. The CSV download includes your edits.")

    rows = result.get("comparison_rows", [])
    columns = ["Product", "Feature", "Specification", "Advantages", "Disadvantages"]
    df = pd.DataFrame(rows, columns=columns).fillna("Not available in document")

    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=True)

    st.download_button(
        label="⬇️ Download Comparison as CSV",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name="product_comparison.csv",
        mime="text/csv",
    )

    # --- Similarities & differences ---------------------------------------------
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("🤝 Similarities")
        show_list(result.get("similarities", []))
    with col_b:
        st.subheader("↔️ Differences")
        show_list(result.get("differences", []))

    # --- Trade-offs ---------------------------------------------------------------
    st.subheader("⚖️ Trade-Offs")
    trade_offs = result.get("trade_offs", [])
    if trade_offs:
        for item in trade_offs:
            with st.expander(f"If you choose: {item.get('product', 'Unknown product')}", expanded=True):
                st.markdown(f"✅ **You gain:** {item.get('you_gain', 'Not available in document')}")
                st.markdown(f"❌ **You lose:** {item.get('you_lose', 'Not available in document')}")
    else:
        st.write("No trade-off information provided.")

    # --- Recommendation --------------------------------------------------------------
    st.subheader("🏆 AI Recommendation")
    rec = result.get("recommendation", {})
    st.markdown(
        f"""<div class="rec-box">
        <h4 style="margin-top:0">Best match: {rec.get('best_product', 'Not determined')}</h4>
        <p>{rec.get('reason', 'No explanation provided.')}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    missing = rec.get("missing_information", [])
    if missing:
        st.markdown("**⚠️ Information not available in the documents:**")
        show_list(missing)

    st.caption("This recommendation is based only on the uploaded documents. "
               "Please double-check important details before buying.")
