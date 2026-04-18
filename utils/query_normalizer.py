# Smart Campus Assistant — Subject Query Normalization & Chunk Scoping
# Extracted from DynamicRouterChain._normalize_subject_query, _scope_chunks_to_subject, _merge_scoped_text

import re
from utils.regex_patterns import SUBJECT_ALIASES, COURSE_CODE


def normalize_subject_query(query, docs):
    """Resolve the subject name from a query using alias dictionary + fuzzy matching.

    Layer 1: Check curated alias dictionary.
    Layer 2: Extract subject names from retrieved chunks and fuzzy match.
    Returns the best-matching subject name (lowercase) or None.
    """
    q_lower = query.lower()

    # Layer 1 — Alias dictionary lookup (longest alias first)
    for alias in sorted(SUBJECT_ALIASES, key=len, reverse=True):
        pattern = r'(?:^|\W)' + re.escape(alias) + r'(?:\W|$)'
        if re.search(pattern, q_lower):
            return SUBJECT_ALIASES[alias]

    # Layer 2 — Extract subject names from chunks and fuzzy match
    candidate_names = set()
    for doc in docs:
        content = doc.page_content
        code_match = COURSE_CODE.search(content)
        if code_match:
            lines_before = content[:code_match.start()].strip().split('\n')
            for line in reversed(lines_before):
                line_clean = line.strip()
                if (line_clean and len(line_clean) > 5
                        and line_clean[0].isupper()
                        and not line_clean.startswith('GOKARAJU')
                        and not line_clean.startswith('---')):
                    candidate_names.add(line_clean.lower())
                    break

    if not candidate_names:
        return None

    # Fuzzy match: score each candidate by token overlap with query
    query_tokens = set(re.findall(r'\w{3,}', q_lower))
    noise = {'what', 'how', 'many', 'the', 'for', 'are', 'is', 'in', 'and',
             'units', 'unit', 'topics', 'topic', 'course', 'code', 'credits',
             'textbooks', 'books', 'references', 'outcomes', 'prerequisites',
             'semester', 'year', 'tell', 'list', 'give', 'show', 'about'}
    query_tokens -= noise

    best_name = None
    best_score = 0
    for name in candidate_names:
        name_tokens = set(re.findall(r'\w{3,}', name))
        if not name_tokens or not query_tokens:
            continue
        overlap = query_tokens & name_tokens
        score = len(overlap) / min(len(query_tokens), len(name_tokens))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= 0.5 and best_name:
        return best_name

    return None


def scope_chunks_to_subject(subject_name, docs):
    """Filter retrieved chunks to only those belonging to the target subject."""
    scoped = []
    subject_lower = subject_name.lower()
    subject_tokens = set(re.findall(r'\w{3,}', subject_lower))

    for doc in docs:
        content_lower = doc.page_content.lower()

        if subject_lower in content_lower:
            scoped.append(doc)
            continue

        content_tokens = set(re.findall(r'\w{3,}', content_lower))
        overlap = subject_tokens & content_tokens
        threshold = max(2, int(len(subject_tokens) * 0.6))
        if len(overlap) >= threshold:
            scoped.append(doc)

    return scoped


def merge_scoped_text(scoped_docs):
    """Concatenate scoped chunks to prevent unit content truncation at chunk boundaries."""
    return "\n\n".join(doc.page_content for doc in scoped_docs)
