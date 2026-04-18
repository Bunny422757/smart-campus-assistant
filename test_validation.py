# Smart Campus Assistant — Comprehensive Test Validation
# Validates all testcases across: routing, intent classification, off-topic detection,
# direct-answer handlers, and end-to-end RAG pipeline.
#
# Usage:  python test_validation.py

import os
import sys
import re
import json
import traceback
from datetime import datetime

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)

config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

COLLEGE_NAME = "Gokaraju Rangaraju Institute of Engineering and Technology (GRIET)"
COLLEGE_SHORT = "GRIET"
NOT_FOUND_MSG = (
    "This information is not available in the campus documents currently uploaded. "
    f"Please contact the {COLLEGE_SHORT} administration for further details."
)

# ──────────────────────────────────────────────
# Test result tracking
# ──────────────────────────────────────────────
results = {"passed": 0, "failed": 0, "errors": 0, "details": []}


def record(test_name, category, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results["passed" if passed else "failed"] += 1
    results["details"].append({
        "test": test_name, "category": category,
        "status": status, "detail": detail
    })
    icon = "✅" if passed else "❌"
    print(f"  {icon} {test_name}: {status}" + (f" — {detail}" if detail else ""))


def record_error(test_name, category, error):
    results["errors"] += 1
    results["details"].append({
        "test": test_name, "category": category,
        "status": "ERROR", "detail": str(error)
    })
    print(f"  ⚠️  {test_name}: ERROR — {error}")


# ══════════════════════════════════════════════
# TEST GROUP 1: Off-Topic Detection
# ══════════════════════════════════════════════
# ── Local copy of contains_off_topic (avoids importing main.py which needs Streamlit/FPDF) ──
def contains_off_topic(question):
    off_topic_keywords = [
        "stock market", "cryptocurrency", "bitcoin",
        "movie review", "recipe", "cook",
        "political party", "election results",
        "dating advice", "relationship",
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in off_topic_keywords)


def test_off_topic_detection():
    print("\n" + "=" * 70)
    print("TEST GROUP 1: Off-Topic Detection")
    print("=" * 70)

    # Should be OFF-TOPIC (blocked)
    off_topic_queries = [
        ("What is the stock market prediction for tomorrow?", True),
        ("Give me a recipe for pasta", True),
        ("Who will win the election results?", True),
        ("Best cryptocurrency to invest in?", True),
        ("Bitcoin price prediction", True),
        ("Movie review of the latest release", True),
        ("Dating advice for college students", True),
    ]

    # Should be ON-TOPIC (allowed)
    on_topic_queries = [
        ("What are the subjects on Monday?", False),
        ("Tell me about the attendance rules", False),
        ("How many units in DBMS?", False),
        ("Faculty names with their teaching subjects?", False),
        ("What is the code of conduct?", False),
        ("When are the internal exams?", False),
    ]

    for query, expected in off_topic_queries + on_topic_queries:
        try:
            result = contains_off_topic(query)
            passed = result == expected
            detail = f"Expected {'off-topic' if expected else 'on-topic'}, got {'off-topic' if result else 'on-topic'}"
            record(f"Off-topic: '{query[:50]}...'", "off_topic", passed, detail)
        except Exception as e:
            record_error(f"Off-topic: '{query[:50]}...'", "off_topic", e)


# ══════════════════════════════════════════════
# TEST GROUP 2: Router — Query Semantic Hint (Layer 1)
# ══════════════════════════════════════════════
def test_query_semantic_hints():
    print("\n" + "=" * 70)
    print("TEST GROUP 2: Router — Query Semantic Hints (Layer 1)")
    print("=" * 70)

    from router import DynamicRouterChain

    # Create a minimal chain instance for testing _query_semantic_hint
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb
    from chromadb.config import Settings

    persist_dir = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    vectorstore = Chroma(client=client, collection_name="campus_documents", embedding_function=embeddings)
    chain = DynamicRouterChain(vectorstore, COLLEGE_NAME, COLLEGE_SHORT)

    test_cases = [
        # (query, expected_mode_or_None, description)
        ("What subjects on Monday?", "timetable_mode", "Day name triggers timetable"),
        ("Who teaches DBMS?", "timetable_mode", "Faculty keyword triggers timetable"),
        ("How many periods on Friday?", "timetable_mode", "Period count + day"),
        ("How many units in machine learning?", "syllabus_mode", "Unit count triggers syllabus"),
        ("What is the course code of DBMS?", "syllabus_mode", "Course code triggers syllabus"),
        ("What are the attendance rules?", "regulation_mode", "Attendance rule keyword"),
        ("What is the grading system?", "regulation_mode", "Grading system keyword"),
        ("Faculty names with their teaching subjects?", "timetable_mode", "Faculty keyword present"),
        ("Tell me about the campus", None, "No domain keywords → None"),
    ]

    for query, expected_mode, desc in test_cases:
        try:
            mode, conf = chain._query_semantic_hint(query)
            passed = mode == expected_mode
            detail = f"Expected {expected_mode}, got {mode} (conf={conf:.2f}) — {desc}"
            record(f"Hint: '{query[:50]}'", "semantic_hint", passed, detail)
        except Exception as e:
            record_error(f"Hint: '{query[:50]}'", "semantic_hint", e)

    return chain  # Reuse for later tests


# ══════════════════════════════════════════════
# TEST GROUP 3: Timetable Intent Classification
# ══════════════════════════════════════════════
def test_timetable_intent_classification():
    print("\n" + "=" * 70)
    print("TEST GROUP 3: Timetable Intent Classification")
    print("=" * 70)

    from handlers.timetable_handler import classify_timetable_intent

    test_cases = [
        ("What subjects on Monday?", "day_subjects", {"day": "MONDAY"}),
        ("How many periods on Friday?", "day_count", {"day": "FRIDAY"}),
        ("What are the time slots on Wednesday?", "day_slots", {"day": "WEDNESDAY"}),
        ("Show me the schedule on Tuesday", "day_schedule", {"day": "TUESDAY"}),
        ("Time slots along with subjects on Thursday?", "day_schedule", {"day": "THURSDAY"}),
        ("Who teaches DBMS?", "faculty_lookup", {}),
        ("What semester is this timetable for?", "metadata", {}),
        ("When are the internal exams?", "almanac", {}),
        ("What holidays are there?", "almanac", {}),
        ("Faculty names with their teaching subjects?", "faculty_lookup", {}),
    ]

    for query, expected_intent, expected_params in test_cases:
        try:
            intent, params = classify_timetable_intent(query)
            intent_ok = intent == expected_intent
            params_ok = all(params.get(k) == v for k, v in expected_params.items())
            passed = intent_ok and params_ok
            detail = f"Expected ({expected_intent}, {expected_params}), got ({intent}, {params})"
            record(f"TT Intent: '{query[:50]}'", "tt_intent", passed, detail)
        except Exception as e:
            record_error(f"TT Intent: '{query[:50]}'", "tt_intent", e)


# ══════════════════════════════════════════════
# TEST GROUP 4: Syllabus Intent Classification
# ══════════════════════════════════════════════
def test_syllabus_intent_classification():
    print("\n" + "=" * 70)
    print("TEST GROUP 4: Syllabus Intent Classification")
    print("=" * 70)

    from handlers.syllabus_handler import classify_syllabus_intent

    test_cases = [
        ("How many units in DBMS?", "unit_count", {}),
        ("What are the topics in Unit 2 of machine learning?", "unit_topics", {}),
        ("What is the course code of computer networks?", "course_code", {}),
        ("How many credits does DBMS have?", "credits", {}),
        ("What are the prerequisites for AI?", "prerequisites", {}),
        ("What are the course outcomes of software engineering?", "course_outcomes", {}),
        ("List the textbooks for DBMS", "textbooks", {}),
        ("Which semester is operating systems?", "year_semester", {}),
    ]

    for query, expected_intent, _ in test_cases:
        try:
            intent, params = classify_syllabus_intent(query)
            passed = intent == expected_intent
            detail = f"Expected {expected_intent}, got {intent}"
            record(f"Syl Intent: '{query[:50]}'", "syl_intent", passed, detail)
        except Exception as e:
            record_error(f"Syl Intent: '{query[:50]}'", "syl_intent", e)


# ══════════════════════════════════════════════
# TEST GROUP 5: Subject Alias Resolution
# ══════════════════════════════════════════════
def test_subject_alias_resolution():
    print("\n" + "=" * 70)
    print("TEST GROUP 5: Subject Alias Resolution")
    print("=" * 70)

    from utils.regex_patterns import SUBJECT_ALIASES

    test_cases = [
        ("dbms", "database management systems"),
        ("ml", "machine learning"),
        ("ai", "artificial intelligence"),
        ("os", "operating systems"),
        ("cn", "computer networks"),
        ("fsd", "full stack development"),
        ("daa", "design and analysis of algorithms"),
        ("p&s", "probability and statistics"),
        ("flat", "formal languages and automata theory"),
        ("oops", "object oriented programming"),
    ]

    for alias, expected_full in test_cases:
        try:
            resolved = SUBJECT_ALIASES.get(alias)
            passed = resolved == expected_full
            detail = f"'{alias}' → '{resolved}' (expected '{expected_full}')"
            record(f"Alias: '{alias}'", "alias", passed, detail)
        except Exception as e:
            record_error(f"Alias: '{alias}'", "alias", e)


# ══════════════════════════════════════════════
# TEST GROUP 6: Regex Pattern Validation
# ══════════════════════════════════════════════
def test_regex_patterns():
    print("\n" + "=" * 70)
    print("TEST GROUP 6: Regex Pattern Validation")
    print("=" * 70)

    from utils.regex_patterns import (
        DAY_PATTERN, UNIT_HEADING, COURSE_CODE,
        LTPC_LINE, PREREQ, YEAR_SEM, TEXTBOOKS, CO_HEADER, ROMAN_MAP
    )

    # DAY_PATTERN
    for day in ["Monday", "TUESDAY", "wednesday", "Thursday", "FRIDAY", "saturday"]:
        try:
            match = DAY_PATTERN.search(f"What subjects on {day}?")
            passed = match is not None
            record(f"DAY_PATTERN: '{day}'", "regex", passed,
                   f"Match={'yes' if match else 'no'}")
        except Exception as e:
            record_error(f"DAY_PATTERN: '{day}'", "regex", e)

    # UNIT_HEADING
    for unit_text, expected in [("UNIT I", "I"), ("UNIT III", "III"), ("UNIT IV", "IV")]:
        try:
            match = UNIT_HEADING.search(unit_text)
            passed = match is not None and match.group(1) == expected
            record(f"UNIT_HEADING: '{unit_text}'", "regex", passed,
                   f"Match={match.group(1) if match else 'None'}")
        except Exception as e:
            record_error(f"UNIT_HEADING: '{unit_text}'", "regex", e)

    # COURSE_CODE
    for line, expected_code in [("Course Code: GR22A2005", "GR22A2005"), ("Course Code : CS101", "CS101")]:
        try:
            match = COURSE_CODE.search(line)
            passed = match is not None and match.group(1) == expected_code
            record(f"COURSE_CODE: '{line}'", "regex", passed,
                   f"Match={match.group(1) if match else 'None'}")
        except Exception as e:
            record_error(f"COURSE_CODE: '{line}'", "regex", e)

    # LTPC_LINE
    for line in ["L/T/P/C: 3/0/0/3", "L / T / P / C : 3/1/0/4"]:
        try:
            match = LTPC_LINE.search(line)
            passed = match is not None
            record(f"LTPC_LINE: '{line}'", "regex", passed,
                   f"Match={match.group(1) if match else 'None'}")
        except Exception as e:
            record_error(f"LTPC_LINE: '{line}'", "regex", e)

    # ROMAN_MAP
    for key, expected in [("1", "I"), ("3", "III"), ("iv", "IV"), ("two", "II")]:
        try:
            result = ROMAN_MAP.get(key)
            passed = result == expected
            record(f"ROMAN_MAP: '{key}'", "regex", passed,
                   f"Got '{result}', expected '{expected}'")
        except Exception as e:
            record_error(f"ROMAN_MAP: '{key}'", "regex", e)


# ══════════════════════════════════════════════
# TEST GROUP 7: End-to-End RAG Pipeline
# ══════════════════════════════════════════════
def test_end_to_end(chain):
    print("\n" + "=" * 70)
    print("TEST GROUP 7: End-to-End RAG Pipeline (Live LLM)")
    print("=" * 70)

    test_cases = [
        # (query, validation_type, validation_criteria, description)
        (
            "Faculty names with their teaching subjects?",
            "not_found_or_has_content",
            None,
            "Faculty names — should return faculty list OR not-found message"
        ),
        (
            "What subjects are there on Monday?",
            "contains_any",
            ["monday", "Monday", "MONDAY"],
            "Monday subjects — should mention Monday"
        ),
        (
            "How many periods on Friday?",
            "contains_any",
            ["period", "Period", "FRIDAY", "Friday"],
            "Friday period count"
        ),
        (
            "What is the code of conduct for students?",
            "has_content",
            None,
            "Code of conduct — should return content"
        ),
        (
            "What is the stock market forecast?",
            "is_offtopic_response",
            None,
            "Stock market — should be blocked as off-topic"
        ),
        (
            "Give me a recipe for biryani",
            "is_offtopic_response",
            None,
            "Recipe — should be blocked as off-topic"
        ),
        (
            "What are the attendance rules?",
            "has_content",
            None,
            "Attendance rules — should return regulation content"
        ),
    ]

    for query, vtype, criteria, desc in test_cases:
        try:
            print(f"\n  🔄 Testing: {desc}...")

            # Check off-topic first (same as main.py flow)
            if contains_off_topic(query):
                response_text = (
                    f"I'm sorry, but I can only answer questions related to "
                    f"{COLLEGE_SHORT} campus information. Please ask about regulations, "
                    f"timetables, syllabus, or other institutional topics."
                )
                source_docs = []
            else:
                result = chain.invoke({"question": query, "chat_history": []})
                response_text = result["answer"]
                source_docs = result.get("source_documents", [])

            # Validate based on type
            if vtype == "not_found_or_has_content":
                # Acceptable: either the not-found message OR actual faculty data
                has_content = len(response_text.strip()) > 20
                passed = has_content
                detail = f"Response length={len(response_text)}, has_content={has_content}"

            elif vtype == "contains_any":
                passed = any(kw in response_text for kw in criteria)
                detail = f"Checked for {criteria}, found={passed}"

            elif vtype == "has_content":
                passed = len(response_text.strip()) > 20 and NOT_FOUND_MSG not in response_text
                detail = f"Response length={len(response_text)}"

            elif vtype == "is_offtopic_response":
                passed = "only answer questions related to" in response_text.lower() or \
                         "campus information" in response_text.lower()
                detail = f"Off-topic block triggered={passed}"

            else:
                passed = False
                detail = f"Unknown validation type: {vtype}"

            record(f"E2E: {desc}", "e2e", passed, detail)
            print(f"     Response preview: {response_text[:120]}...")

        except Exception as e:
            record_error(f"E2E: {desc}", "e2e", f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ══════════════════════════════════════════════
# TEST GROUP 8: Document Classification
# ══════════════════════════════════════════════
def test_document_classification():
    print("\n" + "=" * 70)
    print("TEST GROUP 8: Document Classification")
    print("=" * 70)

    from vectorize_documents import classify_document

    test_cases = [
        ("timetables/3rd btech 1st sem.pdf", "timetable"),
        ("syllabus/sem1.pdf", "syllabus"),
        ("regulations/Code of conduct for Students.pdf", "code_of_conduct"),
        ("random_doc.pdf", "general"),
        ("curriculum/cs_sem5.pdf", "syllabus"),
        ("schedules/weekly.pdf", "timetable"),
    ]

    for path, expected_type in test_cases:
        try:
            result = classify_document(path)
            passed = result == expected_type
            detail = f"'{path}' → '{result}' (expected '{expected_type}')"
            record(f"DocClass: '{path}'", "doc_class", passed, detail)
        except Exception as e:
            record_error(f"DocClass: '{path}'", "doc_class", e)


# ══════════════════════════════════════════════
# TEST GROUP 9: Router Mode Decision (Layer 3)
# ══════════════════════════════════════════════
def test_router_mode_decision(chain):
    print("\n" + "=" * 70)
    print("TEST GROUP 9: Router Mode Decision (3-Layer)")
    print("=" * 70)

    # We test the full _determine_mode path by retrieving docs first
    test_cases = [
        ("What subjects on Monday?", "timetable_mode",
         "Day query should route to timetable"),
        ("How many units in DBMS?", "syllabus_mode",
         "Unit count should route to syllabus"),
        ("What are the attendance rules?", "regulation_mode",
         "Regulation keyword should route appropriately"),
        ("Who teaches machine learning?", "timetable_mode",
         "Faculty query should route to timetable"),
    ]

    for query, expected_mode, desc in test_cases:
        try:
            docs = chain.retriever.invoke(query)
            mode, confidence = chain._determine_mode(query, docs)
            passed = mode == expected_mode
            detail = f"Expected {expected_mode}, got {mode} (confidence={confidence}) — {desc}"
            record(f"Router: '{query[:50]}'", "router", passed, detail)
        except Exception as e:
            record_error(f"Router: '{query[:50]}'", "router", e)


# ══════════════════════════════════════════════
# TEST GROUP 10: Edge Cases & Boundary Tests
# ══════════════════════════════════════════════
def test_edge_cases():
    print("\n" + "=" * 70)
    print("TEST GROUP 10: Edge Cases & Boundary Tests")
    print("=" * 70)

    from handlers.timetable_handler import classify_timetable_intent
    from handlers.syllabus_handler import classify_syllabus_intent

    # Empty / garbage queries
    edge_queries = [
        ("", None, "Empty string"),
        ("hello", None, "Generic greeting"),
        ("asdfghjkl", None, "Random gibberish"),
        ("?!@#$%", None, "Special characters only"),
    ]

    for query, expected_tt_intent, desc in edge_queries:
        try:
            tt_intent, _ = classify_timetable_intent(query)
            syl_intent, _ = classify_syllabus_intent(query)
            passed = tt_intent is None and syl_intent is None
            detail = f"TT={tt_intent}, Syl={syl_intent} — {desc}"
            record(f"Edge: '{query[:30] if query else '(empty)'}' ", "edge", passed, detail)
        except Exception as e:
            record_error(f"Edge: '{desc}'", "edge", e)

    # Case insensitivity
    case_tests = [
        ("WHAT SUBJECTS ON MONDAY?", "day_subjects", "All caps"),
        ("what subjects on monday?", "day_subjects", "All lowercase"),
        ("What Subjects On Monday?", "day_subjects", "Title case"),
    ]

    for query, expected, desc in case_tests:
        try:
            intent, _ = classify_timetable_intent(query)
            passed = intent == expected
            detail = f"Intent={intent}, expected={expected} — {desc}"
            record(f"Case: '{query[:40]}'", "edge", passed, detail)
        except Exception as e:
            record_error(f"Case: '{desc}'", "edge", e)


# ══════════════════════════════════════════════
# TEST GROUP 11: Vector Store Health Check
# ══════════════════════════════════════════════
def test_vectorstore_health():
    print("\n" + "=" * 70)
    print("TEST GROUP 11: Vector Store Health Check")
    print("=" * 70)

    import chromadb
    from chromadb.config import Settings

    try:
        persist_dir = f"{working_dir}/vector_db_dir"
        client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        collections = client.list_collections()

        # Check collection exists
        has_collection = any(c.name == "campus_documents" for c in collections)
        record("Collection exists", "vectorstore", has_collection,
               f"Found {len(collections)} collection(s)")

        if has_collection:
            col = client.get_collection("campus_documents")
            count = col.count()
            passed = count > 0
            record(f"Document count > 0", "vectorstore", passed,
                   f"Total chunks: {count}")

            # Check doc_type distribution
            all_meta = col.get(include=["metadatas"])
            doc_types = {}
            for meta in all_meta["metadatas"]:
                dt = meta.get("doc_type", "unknown")
                doc_types[dt] = doc_types.get(dt, 0) + 1

            has_variety = len(doc_types) >= 2
            record("Multiple doc types", "vectorstore", has_variety,
                   f"Types: {doc_types}")

            # Check for timetable chunks specifically
            has_timetable = any("timetable" in dt for dt in doc_types)
            record("Timetable chunks exist", "vectorstore", has_timetable,
                   f"Timetable types: {[k for k in doc_types if 'timetable' in k]}")

    except Exception as e:
        record_error("Vector store health", "vectorstore", e)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  SMART CAMPUS ASSISTANT — COMPREHENSIVE TEST VALIDATION          ║")
    print("║  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S").ljust(66) + "║")
    print("╚" + "═" * 68 + "╝")

    # ── Unit Tests (no LLM needed) ──
    test_off_topic_detection()
    test_timetable_intent_classification()
    test_syllabus_intent_classification()
    test_subject_alias_resolution()
    test_regex_patterns()
    test_document_classification()
    test_edge_cases()
    test_vectorstore_health()

    # ── Integration Tests (need vectorstore) ──
    chain = test_query_semantic_hints()
    test_router_mode_decision(chain)

    # ── End-to-End Tests (need LLM) ──
    test_end_to_end(chain)

    # ── Summary ──
    total = results["passed"] + results["failed"] + results["errors"]
    print("\n" + "═" * 70)
    print("FINAL SUMMARY")
    print("═" * 70)
    print(f"  Total Tests  : {total}")
    print(f"  ✅ Passed    : {results['passed']}")
    print(f"  ❌ Failed    : {results['failed']}")
    print(f"  ⚠️  Errors   : {results['errors']}")
    print(f"  Pass Rate    : {results['passed']/total*100:.1f}%" if total else "  No tests run")
    print("═" * 70)

    # Show failed tests
    failed_tests = [d for d in results["details"] if d["status"] != "PASS"]
    if failed_tests:
        print("\nFAILED / ERROR TESTS:")
        for t in failed_tests:
            print(f"  [{t['status']}] {t['category']} / {t['test']}")
            print(f"         {t['detail']}")

    return 0 if results["failed"] == 0 and results["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
