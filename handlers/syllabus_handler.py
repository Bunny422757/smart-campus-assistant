# Smart Campus Assistant — Syllabus Direct-Answer Handler
# Extracted from DynamicRouterChain._classify_syllabus_intent and _try_syllabus_direct_answer

import re
from utils.regex_patterns import (
    UNIT_HEADING, COURSE_CODE, LTPC_LINE, PREREQ,
    YEAR_SEM, TEXTBOOKS, CO_HEADER, ROMAN_MAP,
)
from utils.query_normalizer import (
    normalize_subject_query, scope_chunks_to_subject, merge_scoped_text,
)


def classify_syllabus_intent(question):
    """Classify a syllabus query into a specific intent category.

    Returns (intent, params) where intent is one of:
      unit_count, unit_topics, course_code, credits,
      prerequisites, course_outcomes, textbooks, year_semester, or None.
    """
    q = question.lower()

    if re.search(r'how many\s+units', q):
        return ("unit_count", {})

    unit_num_match = re.search(
        r'unit\s+(\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight)',
        q, re.IGNORECASE
    )
    if unit_num_match and re.search(r'topic|about|content|syllabus|cover', q):
        return ("unit_topics", {"unit_ref": unit_num_match.group(1)})

    if re.search(r'course\s*code|subject\s*code', q):
        return ("course_code", {})

    if re.search(r'credit|l/?t/?p/?c|lecture\s*hours?|practical\s*hours?', q):
        return ("credits", {})

    if re.search(r'pre-?\s*requisite', q):
        return ("prerequisites", {})

    if re.search(r'course\s*outcome|\bcos?\b', q):
        return ("course_outcomes", {})

    if re.search(r'text\s*book|reference|books?\s+for|books?\s+of', q):
        return ("textbooks", {})

    if re.search(r'which\s+(year|sem)|what\s+(year|sem)', q):
        return ("year_semester", {})

    return (None, {})


def try_syllabus_direct_answer(question, docs):
    """Attempt to answer structured syllabus queries directly from
    retrieved chunks without calling the LLM.

    Pipeline: classify -> normalize subject -> scope chunks -> merge -> extract.
    Returns the answer string if successful, or None to fall through to the LLM.
    """
    intent, params = classify_syllabus_intent(question)
    if intent is None:
        return None

    # Step 1: Normalize subject from query
    subject = normalize_subject_query(question, docs)
    if not subject:
        print(f"[SYLLABUS] Intent: {intent} | Could not identify subject, falling to LLM")
        return None

    print(f"[SYLLABUS] Intent: {intent} | Subject: {subject}")

    # Step 2: Scope chunks to subject
    scoped = scope_chunks_to_subject(subject, docs)
    if not scoped:
        print(f"[SYLLABUS] No chunks matched subject '{subject}', falling to LLM")
        return None

    # Step 3: Merge scoped chunks
    merged = merge_scoped_text(scoped)
    subject_title = subject.title()

    # ── Unit count ──
    if intent == "unit_count":
        units = UNIT_HEADING.findall(merged)
        unique_units = list(dict.fromkeys(u.upper() for u in units))
        if unique_units:
            return f"**{subject_title}** has **{len(unique_units)}** units ({', '.join('Unit ' + u for u in unique_units)})."
        return None

    # ── Unit topics ──
    if intent == "unit_topics":
        unit_ref = params.get("unit_ref", "").lower()
        target_roman = ROMAN_MAP.get(unit_ref, unit_ref.upper())

        pattern = re.compile(
            r'UNIT\s+' + re.escape(target_roman) + r'\b(.+?)(?=UNIT\s+[IVXLC]|Text\s*Books?|$)',
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(merged)
        if match:
            content = match.group(1).strip()
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            if lines:
                title = lines[0]
                body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                result = f"**Unit {target_roman} — {title}**"
                if body:
                    result += f"\n\n{body}"
                return result
        return None

    # ── Course code ──
    if intent == "course_code":
        match = COURSE_CODE.search(merged)
        if match:
            return f"Course Code for **{subject_title}**: **{match.group(1)}**"
        return None

    # ── Credits ──
    if intent == "credits":
        match = LTPC_LINE.search(merged)
        if match:
            values = match.group(1).split('/')
            labels = ['Lecture (L)', 'Tutorial (T)', 'Practical (P)', 'Credits (C)']
            parts = [f"- {labels[i]}: {values[i]}" for i in range(min(len(values), 4))]
            return f"**{subject_title}** — L/T/P/C: **{match.group(1)}**\n" + "\n".join(parts)
        return None

    # ── Prerequisites ──
    if intent == "prerequisites":
        match = PREREQ.search(merged)
        if match:
            return f"**Prerequisites for {subject_title}:** {match.group(1).strip()}"
        return None

    # ── Course outcomes ──
    if intent == "course_outcomes":
        co_match = CO_HEADER.search(merged)
        if co_match:
            after_header = merged[co_match.end():].strip()
            co_lines = []
            for line in after_header.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.\s', line):
                    co_lines.append(line)
                elif co_lines and line and not UNIT_HEADING.match(line):
                    co_lines[-1] += ' ' + line
                elif co_lines and (not line or UNIT_HEADING.match(line)):
                    break
            if co_lines:
                return f"**Course Outcomes for {subject_title}:**\n" + "\n".join(f"- {co}" for co in co_lines)
        return None

    # ── Textbooks ──
    if intent == "textbooks":
        tb_match = TEXTBOOKS.search(merged)
        if tb_match:
            after_header = merged[tb_match.end():].strip()
            book_lines = []
            for line in after_header.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.\s', line):
                    book_lines.append(line)
                elif book_lines and line and not UNIT_HEADING.match(line):
                    book_lines[-1] += ' ' + line
                elif book_lines and not line:
                    continue
            if book_lines:
                return f"**TextBooks & References for {subject_title}:**\n" + "\n".join(f"- {b}" for b in book_lines)
        return None

    # ── Year / semester ──
    if intent == "year_semester":
        match = YEAR_SEM.search(merged)
        if match:
            return f"**{subject_title}** is offered in **{match.group(1)} Year, {match.group(2)} Semester**."
        return None

    return None
