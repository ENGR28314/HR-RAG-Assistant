# 👩‍💼 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

The application allows users to upload an HR Policy PDF and ask questions about its contents.

Instead of asking an AI model to answer from general knowledge, the application first searches the uploaded HR policy using semantic similarity and FAISS. The most relevant policy sections are then provided to Groq's `openai/gpt-oss-20b` model to generate a grounded answer.

---

## 🚀 Features

- Upload HR Policy PDF
- Extract text from PDF using PyMuPDF
- Preserve PDF page numbers
- Split policy text into overlapping chunks
- Generate semantic embeddings
- Store embeddings in FAISS
- Perform semantic similarity search
- Retrieve relevant policy sections
- Generate grounded answers using Groq
- Use `openai/gpt-oss-20b`
- Display retrieved source pages
- Display similarity scores
- Chat history
- Clear conversation
- Reset uploaded document
- Similarity threshold for retrieval
- Secure Groq API key through Streamlit Secrets
- GitHub-ready
- Streamlit Cloud-ready

---

# 🧠 RAG Architecture

```text
                   HR Policy PDF
                         │
                         ▼
                    PyMuPDF
                         │
                         ▼
                  Text Extraction
                         │
                         ▼
                 Text Cleaning
                         │
                         ▼
                Chunking + Overlap
                         │
                         ▼
             Sentence Transformer
                         │
                         ▼
                    Embeddings
                         │
                         ▼
                       FAISS
                         │
                         │
User Question ───────────┘
       │
       ▼
Question Embedding
       │
       ▼
FAISS Similarity Search
       │
       ▼
Relevant Policy Chunks
       │
       ▼
Groq API
       │
       ▼
openai/gpt-oss-20b
       │
       ▼
Grounded HR Policy Answer
