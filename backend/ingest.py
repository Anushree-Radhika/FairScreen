import re
from io import BytesIO
from typing import Dict, Any, List
import pypdf
import pdfplumber

def clean_text(t: str) -> str:
    """
    Cleans raw extracted text:
    - Strips any HTML tags
    - Drops strange non-ASCII artifacts while preserving standard punctuation
    - Normalizes extra whitespace and multiple newlines
    """
    if not t:
        return ""
    t = re.sub(r"<[^>]+>", " ", str(t))
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()

def extract_text_from_pdf(file_source) -> Dict[str, Any]:
    """
    Extracts plain text and structure from an uploaded PDF (Streamlit UploadedFile, BytesIO, or filepath).
    Uses pypdf as primary fast in-memory parser, with pdfplumber layout-mode fallback.
    """
    if hasattr(file_source, "seek"):
        file_source.seek(0)

    pages_text: List[str] = []
    page_count = 0

    try:
        reader = pypdf.PdfReader(file_source)
        page_count = len(reader.pages)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            pages_text.append(clean_text(extracted))
    except Exception:
        # Fallback to pdfplumber on error
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        with pdfplumber.open(file_source) as pdf:
            page_count = len(pdf.pages)
            pages_text = [clean_text(page.extract_text(layout=True) or "") for page in pdf.pages]

    full_text = "\n\n".join([p for p in pages_text if p.strip()])

    # Sparse text fallback (e.g. multi-column layout missing text in standard extraction)
    if len(full_text.strip()) < 50 and page_count > 0:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            with pdfplumber.open(file_source) as pdf:
                pages_text = [clean_text(page.extract_text(layout=True) or "") for page in pdf.pages]
                full_text = "\n\n".join([p for p in pages_text if p.strip()])
        except Exception:
            pass

    words = full_text.split()
    return {
        "text": full_text,
        "page_count": page_count,
        "pages": pages_text,
        "word_count": len(words),
        "char_count": len(full_text)
    }

def extract_sections(text: str) -> Dict[str, str]:
    """
    Heuristically partitions the resume into standard sections:
    Summary, Skills, Experience, Education, Projects, Certifications.
    """
    section_patterns = {
        "summary": r"(?:summary|objective|professional summary|profile|about me)",
        "skills": r"(?:technical skills|skills|technologies|core competencies|expertise)",
        "experience": r"(?:work experience|experience|employment history|work history|professional experience)",
        "education": r"(?:education|academic background|qualifications)",
        "projects": r"(?:projects|academic projects|key projects)",
        "certifications": r"(?:certifications|certificates|licenses|achievements)"
    }

    headers_regex = r"(?im)^(?:\s*)(?P<header>" + "|".join(section_patterns.values()) + r")(?:\s*[:\-]|\s*$)"
    matches = list(re.finditer(headers_regex, text))

    if not matches:
        return {"full_content": text}

    sections: Dict[str, str] = {}
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections["header_or_contact"] = preamble

    for i, match in enumerate(matches):
        raw_header = match.group("header").strip().lower()
        matched_key = "other"
        for key, pattern in section_patterns.items():
            if re.fullmatch(pattern, raw_header, re.IGNORECASE):
                matched_key = key
                break
        
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[matched_key] = text[start_idx:end_idx].strip()

    return sections
