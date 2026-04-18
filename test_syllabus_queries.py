import sys
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from router import DynamicRouterChain
import chromadb
import logging
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = "gsk_DUMMY_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
logging.getLogger("chromadb").setLevel(logging.ERROR)

def run_tests():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="vector_db_dir")
    vectorstore = Chroma(client=client, collection_name="campus_documents_v3", embedding_function=embeddings)
    router = DynamicRouterChain(vectorstore, "GRIET", "GRIET")
    class DummyLLM:
        def invoke(self, m): return type("R", (), {"content": "[Mocked LLM generation -- retrieval valid]"})()
    router.llm = DummyLLM()
    router._condense_question = lambda q, h: q

    queries = {
        "TEST GROUP 1": [
            "machine learning unit 1 topics",
            "machine learning unit 2 topics",
            "machine learning unit 3 topics",
            "machine learning unit 4 topics"
        ],
        "TEST GROUP 2": [
            "dbms unit 2 topics",
            "software engineering unit 2 topics",
            "computer networks unit 2 topics"
        ],
        "TEST GROUP 3": [
            "ml unit 2",
            "machine learning module 2",
            "topics in ml unit 2",
            "what are the topics in machine learning unit 2"
        ],
        "TEST GROUP 4": [
            "machine learning unit 8",
            "dbms unit 9 topics"
        ]
    }

    for group, qs in queries.items():
        print(f"\n{'='*50}\n{group}\n{'='*50}")
        for q in qs:
            print(f"\nQUERY: {q}")
            try:
                result = router.invoke({"question": q, "chat_history": [], "category": "Syllabus", "section": "All", "semester": "All"})
                
                docs = result.get("source_documents", [])
                
                subjects = set()
                units = set()
                for d in docs:
                    subjects.add(str(d.metadata.get("subject")))
                    units.add(str(d.metadata.get("unit")))
                    
                print(f"RETRIEVED SUBJECT: {', '.join(subjects) if subjects else 'None'}")
                print(f"RETRIEVED UNIT: {', '.join(units) if units else 'None'}")
                if len(subjects) > 1:
                    print("CONTAMINATION STATUS: FAILED (Multiple Subjects)")
                elif len(subjects) == 1:
                    print("CONTAMINATION STATUS: PASS (Clean Subject)")
                else:
                    print("CONTAMINATION STATUS: N/A")
                    
                ans_preview = result["answer"].replace('\n', ' ')
                if len(ans_preview) > 100:
                    ans_preview = ans_preview[:100] + "..."
                print(f"ANSWER PREVIEW: {ans_preview}")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    run_tests()
