import sys
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from main import setup_vectorstore
from router import DynamicRouterChain

def test_pipeline():
    print("\n" + "="*50)
    print(" TIMETABLE FINAL STABILIZATION: PHASE 5 VALIDATION")
    print("="*50)
    
    vs = setup_vectorstore()
    router = DynamicRouterChain(vectorstore=vs)
    
    questions = [
        "What subjects do I have on Monday?",
        "What are the time slots on Monday?",
        "Who teaches machine learning?",
        "What is semester and section?",
        "Faculty names and course codes"
    ]
    
    # We simulate the UI dynamically sending across typical constraints 
    # to test the $and filter boundaries.
    for i, q in enumerate(questions, 1):
        print(f"\n[TEST {i}] QUERY: {q}")
        inputs = {
            "question": q,
            "category": "Timetable",
            "section": "All",
            "semester": "All"
        }
        
        try:
            res = router.invoke(inputs)
            print("ANSWER:")
            print("-" * 30)
            print(res["answer"])
            print("-" * 30)
            print(f"RETRIEVED: {len(res['source_documents'])} Chunks")
            for doc in res["source_documents"][:3]:
                meta = doc.metadata
                print(f" -> [{meta.get('doc_type')}] Day: {meta.get('day')} | Sec: {meta.get('section')} | Sem: {meta.get('semester')}")
        except Exception as e:
            print("ERROR:", str(e))

if __name__ == "__main__":
    test_pipeline()
    print("\n[VALIDATION COMPLETE]")
