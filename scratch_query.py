import os
import re
import random
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading DB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

data = vector_db.get(include=["documents", "metadatas"])
docs = data["documents"]
metas = data["metadatas"]

print(f"Total chunks in DB: {len(docs)}")

print("\n--- 🔍 STEP 1 — CHUNK INSPECTION ---")
random.seed(42)
sample_indices = random.sample(range(len(docs)), min(10, len(docs)))

for idx in sample_indices:
    doc = docs[idx]
    meta = metas[idx]
    # Check headers and keywords
    has_header = bool(re.search(r'Semester: .* \| Subject: .* \| Code: .* \|', doc))
    has_keywords = "Keywords: " in doc
    
    print(f"\nChunk ID: {idx}")
    print(f"Doc Type: {meta.get('doc_type', 'Unknown')}")
    print(f"Has Header: {has_header}")
    print(f"Has Keywords: {has_keywords}")
    print(f"Length: {len(doc)} chars")
    
print("\n--- 🔍 STEP 2 — KEYWORD QUALITY ---")
keyword_chunks = [d for d in docs if "Keywords: " in d]
for d in keyword_chunks[:5]:
    k_line = [l for l in d.split('\n') if l.startswith('Keywords:')][0]
    print(k_line)
    
print("\n--- 🔍 STEP 3 — SUBJECT METADATA VALIDATION ---")
subjects = set()
for m in metas:
    if m and "subject" in m:
        subjects.add(m["subject"])
    elif m and "meta_subject" in m:
        subjects.add(m["meta_subject"])

print(f"Found {len(subjects)} unique subjects:")
for s in sorted(list(subjects)):
    print(f"- {s}")

print("\n--- 🔍 STEP 4 & 6 — RETRIEVAL TESTS ---")
queries = [
    "topics in digital logic design",
    "unit 1 dld",
    "daa unit 3 topics",
    "automata syllabus",
    "subjects on monday",
    "time slots wednesday",
    "section coordinator",
    "faculty teaching machine learning"
]

for q in queries:
    print(f"\nQuery: '{q}'")
    results = vector_db.similarity_search_with_score(q, k=3)
    for r, score in results:
        dtype = r.metadata.get("doc_type", "Unknown")
        subj = r.metadata.get("subject", r.metadata.get("meta_subject", "N/A"))
        print(f"  [{dtype}] (Score: {score:.3f}) Subj: {subj}")

print("\n--- 🔍 STEP 5 — DUPLICATION CHECK ---")
content_counts = defaultdict(int)
duplicates = 0
for d in docs:
    content_counts[d] += 1
    if content_counts[d] > 1:
        duplicates += 1

print(f"Found {duplicates} duplicate chunks.")
