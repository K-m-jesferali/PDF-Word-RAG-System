import os
import faiss
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from docx import Document

# Load embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyPDF2"""
    text = ""
    total_pages = 0
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except Exception as e:
                print(f"⚠️ Skipping page {page_num} in {pdf_path} due to error: {e}")
    return text, total_pages


def extract_text_from_docx(docx_path):
    """Extract text from a Word .docx file"""
    text = ""
    total_pages = 1  # Word files may not have pages; estimate 1 page per 1000 chars
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():  # skip empty lines
                text += para.text + "\n"
        total_pages = max(1, len(text) // 1000 + 1)
    except Exception as e:
        print(f"⚠️ Error reading {docx_path}: {e}")
    return text, total_pages


def files_to_vectors(file_paths):
    all_chunks = []
    all_metadata = []
    total_pages_all = 0

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()
        print(f"📄 Reading file: {file_path}")

        # Extract text and total pages
        if ext == ".pdf":
            text, total_pages = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            text, total_pages = extract_text_from_docx(file_path)
        else:
            print(f"⚠️ Unsupported file type: {ext}, skipping...")
            continue

        total_pages_all += total_pages
        print(f"📊 {file_path} -> {total_pages} pages, {len(text):,} characters")

        # Split text into chunks (~500 chars)
        for i in range(0, len(text), 500):
            chunk_text = text[i:i + 500].strip()
            if not chunk_text:
                continue
            # Estimate page number for chunk
            estimated_page = min((i // (len(text) // max(total_pages, 1))) + 1, total_pages)
            all_chunks.append(chunk_text)
            all_metadata.append({
                'file': os.path.basename(file_path),
                'start_pos': i,
                'estimated_page': estimated_page
            })

    print(f"✂️ Created {len(all_chunks)} chunks from {len(file_paths)} files")

    # Encode in batches to avoid memory issues
    def encode_in_batches(model, texts, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [str(b).strip() for b in batch if str(b).strip()]
            if not batch:
                continue
            try:
                emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Encoding error: {e}")
        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

    print("🔄 Getting embeddings from Hugging Face model...")
    embeddings = encode_in_batches(model, all_chunks, batch_size=8)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    # Save FAISS index and metadata
    faiss.write_index(index, "vectors.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': all_chunks,
            'metadata': all_metadata,
            'total_pages': total_pages_all  # <-- added for ask_questions.py
        }, f)

    print("✅ Vector database created successfully!")
    return embeddings, all_chunks


# Usage
if __name__ == "__main__":
    files = [
        "AI and ML.docx",
        "Deep Learning.docx",
        "DS - Sample Questions (Theory).docx",
        "ml questions (2).docx",
        "Machine-Learning-Questions-and-Answers.pdf"
    ]
    embeddings, chunks = files_to_vectors(files)
    print("\n🎉 Setup complete! Now you can run 'ask_questions.py' to chat with your documents!")
