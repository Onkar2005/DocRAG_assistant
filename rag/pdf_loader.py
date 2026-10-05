"""Step 1: Extract and clean text from a PDF."""
import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Fix common PDF extraction problems."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # join words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)       # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)         # collapse extra spaces
    return text.strip()


def extract_pages(file) -> list[dict]:
    """Return a list of {"page": int, "text": str}. `file` can be a path or file-like object."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": i, "text": text})
    return pages
