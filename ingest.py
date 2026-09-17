"""
ingest.py
---------
Reads all PDFs in --input_dir (e.g. Daraz_Comprehensive_Policies_Guide),
splits their text into overlapping chunks, embeds each chunk with a
sentence-transformers model, and saves a FAISS index + metadata
(id, department, source_file) into --output_dir (default: faiss_index).

Usage:
    python ingest.py --input_dir pdfs --output_dir faiss_index
    python ingest.py --input_dir pdfs --output_dir faiss_index --chunk_size 1000 --chunk_overlap 150

Output folder contents:
    faiss_index/
        index.faiss     -> the FAISS vector index
        metadata.json    -> list of {id, department, source_file, page, chunk_index, text}
"""

import os
import json
import argparse
import re

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Department inference
# ---------------------------------------------------------------------------
# Maps keywords found in a filename (or, as a fallback, chunk text) to a
# department label. Edit/extend this dict to match your real file names.
DEPARTMENT_KEYWORDS = {
    "return": "Returns",
    "refund": "Refunds",
    "delivery": "Delivery",
    "shipping": "Delivery",
    "seller": "Sellers",
    "vendor": "Sellers",
    "payment": "Payments",
    "billing": "Payments",
    "customer": "Customer Support",
    "support": "Customer Support",
    "faq": "Customer Support",
}


def infer_department(filename: str, text_sample: str = "") -> str:
    """Guess a department from the filename first, then a text sample."""
    name = filename.lower()
    for keyword, dept in DEPARTMENT_KEYWORDS.items():
        if keyword in name:
            return dept

    sample = text_sample.lower()
    for keyword, dept in DEPARTMENT_KEYWORDS.items():
        if keyword in sample:
            return dept

    return "General"


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: str):
    """Yield (page_number, text) for every page in a PDF."""
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            yield page_num, text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Simple sliding-window chunker over characters, breaking on spaces."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # try not to cut a word in half
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - chunk_overlap
        if start < 0:
            start = 0

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into a FAISS index.")
    parser.add_argument("--input_dir", type=str, default="pdfs",
                         help="Folder containing the PDF file(s).")
    parser.add_argument("--output_dir", type=str, default="faiss_index",
                         help="Folder to write index.faiss and metadata.json into.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                         help="Max characters per chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=150,
                         help="Character overlap between consecutive chunks.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                         help="sentence-transformers model to use for embeddings.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    pdf_files = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {args.input_dir}")

    print(f"Found {len(pdf_files)} PDF(s) in '{args.input_dir}':")
    for f in pdf_files:
        print(f"  - {f}")

    # -------------------------------------------------------------
    # 1) Extract + chunk all PDFs, building metadata as we go
    # -------------------------------------------------------------
    all_chunks = []      # raw text for embedding
    all_metadata = []    # parallel list of metadata dicts
    global_id = 0

    for filename in pdf_files:
        pdf_path = os.path.join(args.input_dir, filename)
        print(f"\nProcessing: {filename}")

        pages = list(extract_pages(pdf_path))
        if not pages:
            print(f"  [!] No extractable text found, skipping.")
            continue

        # sample first page's text to help department inference if filename is generic
        department = infer_department(filename, pages[0][1] if pages else "")

        chunk_index_in_file = 0
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text, args.chunk_size, args.chunk_overlap):
                all_chunks.append(chunk)
                all_metadata.append({
                    "id": global_id,
                    "department": department,
                    "source_file": filename,
                    "page": page_num,
                    "chunk_index": chunk_index_in_file,
                    "text": chunk,
                })
                global_id += 1
                chunk_index_in_file += 1

        print(f"  -> department: {department}, chunks created: {chunk_index_in_file}")

    if not all_chunks:
        raise RuntimeError("No text chunks were extracted from any PDF. Aborting.")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # -------------------------------------------------------------
    # 2) Create embeddings
    # -------------------------------------------------------------
    print(f"Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding chunks"):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # -------------------------------------------------------------
    # 3) Build FAISS index (IndexFlatL2 wrapped with explicit IDs)
    # -------------------------------------------------------------
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)

    ids = np.array([m["id"] for m in all_metadata], dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    print(f"FAISS index built with {index.ntotal} vectors.")

    # -------------------------------------------------------------
    # 4) Save index + metadata
    # -------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    index_path = os.path.join(args.output_dir, "index.faiss")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved FAISS index to: {index_path}")
    print(f"Saved metadata to:    {metadata_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
