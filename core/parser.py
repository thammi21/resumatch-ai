# core/parser.py
"""
Document parsing for the Job Application Assistant.

Handles:
  - Resume PDF → clean text string
  - JD PDF → clean text string
  - JD plain text → clean text string

Week 1 concept revisited: File I/O, error handling, pathlib
New concept: PyMuPDF (fitz) for PDF text extraction
"""

import pymupdf as fitz  # PyMuPDF — imported as fitz (historical name)
import re
from pathlib import Path


# ── PDF Parsing ───────────────────────────────────────────────────
def parse_pdf(file) -> str:
    """
    Extract text from an uploaded PDF file.

    Args:
        file: Streamlit UploadedFile object OR a file path string/Path

    Returns:
        Clean extracted text as a single string

    Raises:
        ValueError: If PDF is empty or unreadable
        RuntimeError: If PDF parsing fails
    """
    try:
        # Handle both Streamlit uploaded files and file paths
        # Streamlit gives us a BytesIO-like object
        # PyMuPDF can open both file paths and bytes
        if hasattr(file, "read"):
            # Streamlit UploadedFile — read as bytes
            file.seek(0)
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            # File path string or Path object
            doc = fitz.open(str(file))

        # Extract text from every page
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            full_text += page_text + "\n"

        doc.close()

        # Clean the extracted text
        cleaned = clean_text(full_text)

        if not cleaned.strip():
            raise ValueError(
                "PDF appears to be empty or contains only images. "
                "Please use a text-based PDF."
            )

        return cleaned

    except Exception as e:
        # Re-raise our own clean errors
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        # Handle corrupted or invalid PDF
        error_str = str(e).lower()
        if "cannot open" in error_str or "no objects" in error_str:
            raise RuntimeError(
                "Could not read the PDF file. "
                "Make sure it is a valid PDF and not corrupted."
            )
        raise RuntimeError(f"PDF parsing failed: {str(e)}")


# ── Text Cleaning ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean raw extracted text for LLM consumption.

    PDF extraction often produces:
    - Multiple consecutive blank lines
    - Weird whitespace and tab characters
    - Null bytes and special characters
    - Inconsistent line endings

    This function normalises all of that.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove null bytes — crash LLMs if present
    text = text.replace("\x00", "")

    # Normalise line endings — Windows uses \r\n, Unix uses \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tab characters with spaces
    text = text.replace("\t", " ")

    # Remove lines that are just whitespace
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]

    # Collapse more than 2 consecutive blank lines into 2
    # Preserves paragraph structure without excessive whitespace
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove multiple spaces within a line
    text = re.sub(r" {2,}", " ", text)


    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── JD Parsing ────────────────────────────────────────────────────
def parse_jd(jd_input, input_type: str = "text") -> str:
    """
    Parse a job description from text or PDF.

    Args:
        jd_input : str (plain text) OR UploadedFile (PDF)
        input_type: "text" or "pdf"

    Returns:
        Clean JD text string ready for the chain

    Raises:
        ValueError: If JD is too short to be valid
    """
    if input_type == "pdf":
        text = parse_pdf(jd_input)
    else:
        # Plain text — just clean it
        text = clean_text(str(jd_input))

    # Validate minimum length
    # A real JD should have at least 100 words
    word_count = len(text.split())
    if word_count < 50:
        raise ValueError(
            f"Job description seems too short ({word_count} words). "
            f"Please paste the complete job description."
        )

    return text


# ── Resume Parsing ────────────────────────────────────────────────
def parse_resume(resume_file) -> str:
    """
    Parse a resume from an uploaded PDF file.

    Args:
        resume_file: Streamlit UploadedFile object

    Returns:
        Clean resume text string ready for the chain

    Raises:
        ValueError: If resume is too short
    """
    text = parse_pdf(resume_file)

    # Validate minimum length
    word_count = len(text.split())
    if word_count < 50:
        raise ValueError(
            f"Resume seems too short ({word_count} words). "
            f"Make sure you uploaded a text-based PDF, not a scanned image."
        )

    return text


# ── Token estimation ──────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for a text string.
    1 token ≈ 0.75 words in English.
    Used to warn user if document is too large for context window.

    Args:
        text: Input string

    Returns:
        Estimated token count
    """
    return int(len(text.split()) / 0.75)


def truncate_if_needed(text: str, max_tokens: int = 3000) -> str:
    """
    Truncate text if it exceeds max_tokens estimate.
    Prevents context window overflow for very long resumes/JDs.

    Truncates at word boundary — never mid-word.

    Args:
        text     : Input string
        max_tokens: Maximum allowed tokens (default 3000)

    Returns:
        Original text if within limit, truncated text otherwise
    """
    estimated = estimate_tokens(text)

    if estimated <= max_tokens:
        return text

    # Calculate how many words to keep
    max_words = int(max_tokens * 0.75)
    words = text.split()
    truncated = " ".join(words[:max_words])

    print(
        f"  ⚠️  Document truncated: {estimated} tokens → {max_tokens} tokens. "
        f"Some content may be cut off."
    )

    return truncated + "\n\n[Document truncated due to length]"


# ── Text input validation ─────────────────────────────────────────
def validate_inputs(resume_text: str, jd_text: str) -> tuple[bool, str]:
    """
    Validate both inputs before running the pipeline.
    Returns (is_valid, error_message).

    Args:
        resume_text: Parsed resume string
        jd_text    : Parsed JD string

    Returns:
        Tuple of (True, "") if valid
        Tuple of (False, error_message) if invalid
    """
    if not resume_text or not resume_text.strip():
        return False, "Resume is empty. Please upload a valid PDF."

    if not jd_text or not jd_text.strip():
        return False, "Job description is empty. Please provide a JD."

    resume_words = len(resume_text.split())
    jd_words = len(jd_text.split())

    if resume_words < 50:
        return False, (
            f"Resume too short ({resume_words} words). "
            f"Please upload your complete resume."
        )

    if jd_words < 30:
        return False, (
            f"Job description too short ({jd_words} words). "
            f"Please paste the complete JD."
        )

    # Warn about very long documents
    resume_tokens = estimate_tokens(resume_text)
    jd_tokens = estimate_tokens(jd_text)

    if resume_tokens > 4000:
        return False, (
            f"Resume is very long ({resume_tokens} estimated tokens). "
            f"Consider using a 1-2 page resume for best results."
        )

    return True, ""