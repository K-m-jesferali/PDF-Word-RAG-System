# PDF & Word RAG System

A **Retrieval-Augmented Generation (RAG) system** that allows you to ask questions and get answers from PDF and Word (.docx) documents using embeddings and a small LLM. This project is built with Python, FAISS, Hugging Face Transformers, and Sentence Transformers.

## Overview

This system allows users to:

1. Convert **PDF** and **Word documents** into vector embeddings.
2. Store embeddings in a **FAISS vector database** for fast retrieval.
3. Ask questions in natural language and get answers **based on the content of the documents**.

It uses a **SentenceTransformer** embedding model to vectorize text and a **lightweight Hugging Face LLM** for generating answers.

---

## Features

- ✅ Supports **PDF (.pdf)** and **Word (.docx)** documents.
- ✅ Splits documents into **manageable text chunks** for embeddings.
- ✅ Stores embeddings in **FAISS** for fast similarity search.
- ✅ Answers questions using **contextual document retrieval**.
- ✅ Lightweight and can run on **CPU or GPU**.
- ✅ Easily extensible to more file formats.

---

## Architecture

1. **Document Processing**  
   - Extract text from PDF using `PyPDF2`.  
   - Extract text from Word using `python-docx`.  
   - Split text into **chunks of 500 characters** (configurable).

2. **Vectorization**  
   - Use `SentenceTransformer` to generate embeddings.  
   - Store embeddings and metadata in **FAISS**.

3. **Question Answering**  
   - Retrieve top-k similar chunks using FAISS.  
   - Build a context from these chunks.  
   - Generate answer using a **small Hugging Face model** (`google/flan-t5-small`).

---

Clone the repository:

```bash
git clone https://github.com/K-m-jesferali/PDF-Word-RAG-System.git
cd PDF-Word-RAG-System


