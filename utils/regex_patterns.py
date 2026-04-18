# Smart Campus Assistant — Shared Regex Patterns & Constants
# Extracted from DynamicRouterChain class attributes for cross-module reuse.

import re

# ──────────────────────────────────────────────
# Timetable patterns
# ──────────────────────────────────────────────

DAY_PATTERN = re.compile(
    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    re.IGNORECASE
)

# ──────────────────────────────────────────────
# Syllabus patterns (GRIET PDF structure)
# ──────────────────────────────────────────────

UNIT_HEADING = re.compile(r'UNIT\s+(I{1,3}V?I{0,2})\b', re.IGNORECASE)
COURSE_CODE = re.compile(r'Course\s*Code\s*:\s*(\S+)', re.IGNORECASE)
LTPC_LINE = re.compile(r'L\s*/\s*T\s*/\s*P\s*/\s*C\s*:\s*([\d/]+)', re.IGNORECASE)
PREREQ = re.compile(r'Pre-?\s*requisites?\s*:\s*(.+)', re.IGNORECASE)
YEAR_SEM = re.compile(r'(I{1,4}V?)\s+Year\s+(I{1,2}V?)\s+Semester', re.IGNORECASE)
TEXTBOOKS = re.compile(r'Text\s*Books?\s*(?:and|&)?\s*References?\s*:', re.IGNORECASE)
CO_HEADER = re.compile(r'Course\s+Outcomes?\s*:?', re.IGNORECASE)

# ──────────────────────────────────────────────
# Subject alias dictionary
# ──────────────────────────────────────────────

SUBJECT_ALIASES = {
    "dbms": "database management systems",
    "p&s": "probability and statistics",
    "prob": "probability and statistics",
    "ps": "probability and statistics",
    "dld": "digital logic design",
    "dsd": "digital system design",
    "os": "operating systems",
    "cn": "computer networks",
    "se": "software engineering",
    "coa": "computer organization and architecture",
    "oops": "object oriented programming",
    "oop": "object oriented programming",
    "daa": "design and analysis of algorithms",
    "ds": "data structures",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "fsd": "full stack development",
    "cd": "compiler design",
    "cc": "cloud computing",
    "wt": "web technologies",
    "es": "embedded systems",
    "sp": "system programming",
    "mp": "microprocessors",
    "flat": "formal languages and automata theory",
    "toc": "theory of computation",
    "dwm": "data warehousing and mining",
    "mefa": "managerial economics and financial analysis",
    "de": "differential equations",
    "la": "linear algebra",
    "dm": "discrete mathematics",
    "aca": "advanced computer architecture",
    "stqa": "software testing and quality assurance",
    "big data": "big data analytics",
    "iot": "internet of things",
    "nlp": "natural language processing",
    "ir": "information retrieval",
    "is": "information security",
    "cns": "cryptography and network security",
}

# ──────────────────────────────────────────────
# Roman numeral map (for unit number conversion)
# ──────────────────────────────────────────────

ROMAN_MAP = {
    '1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V',
    'one': 'I', 'two': 'II', 'three': 'III', 'four': 'IV', 'five': 'V',
    'i': 'I', 'ii': 'II', 'iii': 'III', 'iv': 'IV', 'v': 'V',
    'vi': 'VI', 'vii': 'VII', 'viii': 'VIII',
}
