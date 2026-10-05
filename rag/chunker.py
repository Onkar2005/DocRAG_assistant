"""Step 2: Split text into small overlapping chunks."""


def chunk_pages(pages: list[dict], chunk_size: int = 200, overlap: int = 40) -> list[dict]:
    """Split each page into chunks of ~chunk_size words.

    Overlap keeps sentences that fall on a boundary from losing their context.
    Each chunk remembers its page number so we can show sources.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            if len(piece) < 20 and start > 0:   # skip tiny leftover tails
                continue
            chunks.append({"page": page["page"], "text": " ".join(piece)})
            if start + chunk_size >= len(words):
                break
    return chunks
