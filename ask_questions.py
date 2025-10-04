import os
import faiss
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from docx import Document  # Word document support

# Device detection for Transformers
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

# Load embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("🟢 Embedding model loaded successfully!")

# Load LLM pipeline for Q&A
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # Small and free model
    device_map="auto",
    dtype="auto"
)
print("🟢 LLM pipeline initialized!")

# Function to ask a question
def ask_question(question, top_k=3):
    # Check if database exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Error: Vector database not found! Run 'pdf_to_word_vectors.py' first.")
        return None

    try:
        # Load FAISS index and chunks
        index = faiss.read_index("vectors.index")
        with open("chunks.pkl", "rb") as f:
            data = pickle.load(f)

        chunks = data['chunks']
        metadata = data['metadata']
        total_pages = data['total_pages']

        # Generate embedding for the question
        query_vector = model.encode([question], convert_to_numpy=True)
        scores, indices = index.search(query_vector.astype('float32'), top_k)

        # Retrieve top-k context chunks
        context_parts = []
        for idx in indices[0]:
            chunk_text = chunks[idx]
            page_num = metadata[idx]['estimated_page']
            context_parts.append(f"[Page {page_num}]: {chunk_text}")

        context = "\n\n".join(context_parts)

        # Build prompt for LLM
        prompt = (
            f"You are a helpful assistant answering questions about a {total_pages}-page document.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer using ONLY the context above."
        )

        # Generate answer
        response = qa_pipeline(prompt, do_sample=True)[0]["generated_text"]
        return response.strip()

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return None


# ------------------------------
# Interactive loop
# ------------------------------
def main():
    # Check if database exists
    if not os.path.exists("vectors.index") or not os.path.exists("chunks.pkl"):
        print("❌ Vector database not found! Run 'pdf_to_word_vectors.py' first.")
        return

    # Load database for info
    index = faiss.read_index("vectors.index")
    with open("chunks.pkl", "rb") as f:
        data = pickle.load(f)
    chunks = data['chunks']
    total_pages = data['total_pages']

    print(f"✅ Database loaded: {len(chunks)} chunks from {total_pages} pages")
    print("\n" + "=" * 60)
    print("🤖 RAG System Ready! Ask me questions about your documents")
    print("💡 Type 'bye', 'quit', 'exit', or 'q' to exit")
    print("🔢 Type 'info' to see database statistics")
    print("=" * 60)

    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ["bye", "quit", "exit", "q"]:
            print("👋 Goodbye! Thanks for using the RAG system!")
            break

        if question.lower() == "info":
            print(f"📊 Database Info:")
            print(f"   • Total pages: {total_pages}")
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Vector dimensions: {index.d}")
            print(f"   • Average chunks per page: {len(chunks)/total_pages:.1f}")
            print(f"   • Sample chunk: {chunks[0][:100]}...")
            continue

        if not question:
            print("⚠️  Please enter a question!")
            continue

        print("🔍 Searching and generating answer...")
        answer = ask_question(question)
        if answer:
            print(f"\n🤖 Answer:\n{answer}")
        else:
            print("❌ Could not generate an answer. Try again.")


if __name__ == "__main__":
    main()
