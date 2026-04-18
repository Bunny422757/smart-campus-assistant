# Smart Campus Assistant

## Overview

Smart Campus Assistant is an AI-powered academic query system designed to help students quickly access important campus-related information such as syllabus details, academic regulations, class timetables, notices, and placement information.

The system uses Retrieval-Augmented Generation (RAG) with Llama 3.1, LangChain, and ChromaDB to retrieve accurate information from institutional documents and provide precise, context-aware responses through a user-friendly chat interface.

This project reduces manual searching through PDFs and improves student access to academic resources using intelligent document retrieval and semantic search.

---

## Features

- AI-powered academic query answering
- Syllabus retrieval by subject, unit, textbooks, references, and topics
- Timetable retrieval by day, faculty, and subject
- Academic regulations and code of conduct query handling
- Support for notices and placement-related documents
- Confidence-based subject detection for syllabus queries
- Query intent classification for syllabus routing
- Dynamic routing for different document categories
- Semantic PDF parsing and structured chunking
- Vector database retrieval using ChromaDB
- Streamlit-based interactive chat interface
- FastAPI backend support
- Chat export to PDF

---

## Tech Stack

### Programming Language

- Python

### Frontend

- Streamlit

### Backend

- FastAPI
- Uvicorn
- Pydantic

### AI / LLM Stack

- LangChain
- Llama 3.1
- Groq API

### Vector Database

- ChromaDB
- HuggingFace Embeddings
- sentence-transformers/all-MiniLM-L6-v2

### Document Processing

- PyPDF2
- pdfplumber
- pytesseract (OCR)
- FPDF

### NLP & Processing

- NLTK
- Regular Expressions
- Semantic Chunking

---

## Project Architecture

User Query
↓
Query Classification (Syllabus / Timetable / Regulations / Notices / Placements)
↓
Dynamic Router Chain
↓
Document Retrieval from ChromaDB
↓
Relevant Context Extraction
↓
Llama 3.1 via Groq API
↓
Final Answer Generation
↓
Streamlit UI / FastAPI Response

---

## Folder Structure

```text
smart-campus-assistant/
│
├── data/
│   ├── syllabus/
│   ├── regulations/
│   ├── timetables/
│   ├── notices/
│   └── placements/
│
├── handlers/
│   ├── syllabus_handler.py
│   ├── timetable_handler.py
│   └── regulation_handler.py
│
├── prompts/
│   ├── syllabus_prompt.py
│   ├── timetable_prompt.py
│   └── regulation_prompt.py
│
├── utils/
│   ├── query_normalizer.py
│   └── regex_patterns.py
│
├── main.py
├── router.py
├── vectorize_documents.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone <your-repository-link>
cd smart-campus-assistant
```

### Create Virtual Environment

```bash
conda create -n sca python=3.10
conda activate sca
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Make sure `.env` is included in `.gitignore`.

---

## Run Vector Database Creation

Before starting the application:

```bash
python vectorize_documents.py
```

This will:

- Process PDFs and text files
- Create semantic chunks
- Generate embeddings
- Store vectors in ChromaDB

---

## Run the Application

### Streamlit UI

```bash
streamlit run main.py
```

### FastAPI Backend

```bash
uvicorn main:app --reload
```

---

## Example Queries

### Syllabus

- syllabus of DBMS
- unit 3 of Machine Learning
- textbooks for DBMS
- topics in Artificial Intelligence

### Timetable

- subjects on Monday
- faculty names
- timetable for Thursday
- who teaches DBMS

### Regulations

- attendance rules
- backlog policy
- promotion criteria
- code of conduct

---

## Future Improvements

- Role-based access for students and faculty
- Voice-based query interaction
- Multi-language support
- Department-specific personalization
- Mobile app integration
- Notice automation and placement alert system
- Admin dashboard for document management

---

## Author

Akkinapalli Sai Pranav
B.Tech – Information Technology
Gokaraju Rangaraju Institute of Engineering and Technology (GRIET)

---

## License

This project is developed for academic and educational purposes.
