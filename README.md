# Enterprise Policy Compliance Auditor

An AI-powered RAG (Retrieval-Augmented Generation) application that helps organizations audit internal policy documents for compliance. Upload any policy PDF, and the system indexes it into a vector database so you can ask natural language compliance questions — with answers grounded strictly in the uploaded document.

## Problem

Compliance teams spend hours manually reviewing lengthy policy documents to answer specific questions. Misinterpretations or overlooked clauses can lead to regulatory risk. There is a need for a tool that provides fast, accurate, source-backed answers from policy documents.

## Solution

This application uses RAG to retrieve relevant sections from an uploaded policy PDF and generates answers using NVIDIA NIM (LLaMA 3.1 70B). The AI is constrained to answer only from the document — it will not speculate or use outside knowledge.

## Features

- **PDF Upload & Indexing** — Upload any policy PDF and index it into ChromaDB with one click.
- **Compliance Q&A** — Ask natural language questions and get answers grounded in the document.
- **Source Snippets** — Every answer includes the exact passages used, with page numbers.
- **Strict Grounding** — The AI refuses to answer if the document doesn't contain relevant information.
- **Persistent Vector Store** — Indexed documents are stored locally for repeated querying.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | NVIDIA NIM (LLaMA 3.1 70B Instruct) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Framework | LangChain |
| PDF Parsing | PyPDF |

## Architecture

```
User uploads PDF
    → PyPDFLoader extracts text
    → RecursiveCharacterTextSplitter creates chunks
    → HuggingFace embeddings encode chunks
    → ChromaDB stores vectors

User asks question
    → ChromaDB retrieves top-4 relevant chunks
    → LangChain builds prompt with context
    → NVIDIA NIM generates grounded answer
    → Streamlit displays answer + source snippets
```

## Setup

1. **Clone the repository**
   ```bash
   cd enterprise-compliance-rag
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com).

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Example Compliance Questions

- "What is the company's policy on remote work?"
- "Are there any restrictions on sharing confidential data with third parties?"
- "What is the disciplinary process for policy violations?"
- "Does the policy require annual compliance training?"
- "What are the data retention requirements?"

## Future Improvements

- Multi-document support for comparing policies side by side
- Chat history to maintain conversation context
- Export compliance reports as PDF
- Role-based access control for team use
- Support for additional document formats (DOCX, HTML)
- Automated compliance checklist generation
