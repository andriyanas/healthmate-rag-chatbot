# 🩺 HealthMate — RAG Health Q&A Engine

An enterprise-grade Retrieval-Augmented Generation (RAG) conversational agent engineered for general healthcare inquiry processing. Built leveraging local vector embeddings and the Google Gemini API for cost-efficient response synthesis.

---

## 📌 Architectural Overview

HealthMate employs a **hybrid RAG pipeline** designed to eliminate external API costs during document retrieval while maintaining context fidelity.

```text
+-----------------------------------------------------------------------+
|                            USER QUERY                                 |
+-----------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------+
|  [1] LOCAL SEMANTIC RETRIEVAL                                         |
|  Engine : ChromaDB + SentenceTransformers (all-MiniLM-L6-v2)          |
|  Scope  : Queries top 4 documents from 47,457 indexed records         |
+-----------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------+
|  [2] CONTEXT CONTEXTUALIZATION & PROMPT ASSEMBLY                      |
|  Builds formatted prompt payload with domain rules, top-k context,    |
|  and last 6 conversation turns.                                       |
+-----------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------+
|  [3] RESPONSE GENERATION                                              |
|  API    : Google Gemini API (gemini-3.1-flash-lite)                   |
+-----------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------+
|                    FINAL OUTPUT & CITED SOURCES                       |
+-----------------------------------------------------------------------+
```

### Design Rationale: Cost & Token Efficiency
* **Zero-Cost Indexing & Retrieval**: Vector embeddings and semantic searches execute entirely on local hardware using `sentence-transformers`. The Gemini API is not invoked during data ingestion or initial document retrieval.
* **Bounded Context Injection**: By passing only the top 4 retrieved documents (rather than entire knowledge bases), context windows remain lightweight, minimizing token usage and latency per request.

---

## 📊 Dataset Specifications

* **Source**: [MedQuAD (Hugging Face Repository)](https://huggingface.co/datasets/lavita/MedQuAD)
* **Volume**: 47,457 question-answer pairs curated from 12 official National Institutes of Health (NIH) domains.
* **Scope**: Covers 37 distinct clinical domains, including disease prevention, symptoms, diagnosis, and therapeutic options.
* **Compliance**: Non-public domain entries (e.g., MedlinePlus Drugs/Herbs, A.D.A.M.) are excluded in compliance with distribution standards.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | Google Gemini API (`gemini-3.1-flash-lite`) | Generative answer execution |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local dense vector representations |
| **Vector Database** | ChromaDB | Local vector indexing and similarity search |
| **User Interface** | Streamlit | Web client framework |
| **Data Ingestion** | Datasets (Hugging Face) | Automated pipeline retrieval |

---

## 📁 Repository Structure
```text
healthmate-rag-chatbot/
├── app.py                  # Main Streamlit web application entry point
├── requirements.txt        # System dependency manifestations
├── .env.example            # Environment variable template
├── scripts/
│   ├── download_data.py    # Pipeline to fetch and validate the raw dataset
│   └── build_index.py      # Script for batch vector indexing
├── src/
│   ├── config.py           # Centralized configuration and parameters
│   ├── vector_store.py     # Local vector database handler
│   ├── rag_engine.py       # Orchestrator for retrieval and Gemini API
│   └── prompts.py          # System instructions and prompt templates
└── screenshots/            # UI execution artifacts for documentation
```

## 🚀 Installation & Execution

### 1. Repository Setup & Environment Provisioning

Bash
git clone <repository-url>
cd healthmate-rag-chatbot
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 2. Environment Configuration
Obtain an API key from Google AI Studio. Create a .env file in the root directory:

Bash
cp .env.example .env
Populate the variable:

Code snippet
GEMINI_API_KEY=your_api_key_here

### 3. Data Retrieval
Download and clean the raw MedQuAD dataset:

Bash
python scripts/download_data.py

### 4. Vector Store Ingestion
Build the persistent ChromaDB index locally (Estimated execution time: 10–25 minutes depending on CPU architecture):

Bash
python scripts/build_index.py

### 5. Application Launch
Execute the Streamlit interface:

Bash
streamlit run app.py

## ⚠️ Medical & Legal Disclaimer
HealthMate is strictly designed as an educational demonstration and research application. It does not provide medical diagnoses, treatment recommendations, or clinical decision support. All content is generated using public domain datasets and must not replace professional clinical advice.