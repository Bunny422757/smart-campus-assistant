# Smart Campus Assistant — Timetable Direct-Answer Handler
# Extracted from DynamicRouterChain._classify_timetable_intent and _try_timetable_direct_answer

import re
from utils.regex_patterns import DAY_PATTERN


def classify_timetable_intent(question):
    """Classify a timetable query into a specific intent category.

    Returns (intent, params) where intent is one of:
      day_subjects, day_slots, day_schedule, day_count,
      faculty_lookup, metadata, almanac, or None.
    """
    q = question.lower()

    # Detect target day
    day_match = DAY_PATTERN.search(q)
    day = day_match.group(1).upper() if day_match else None

    # ── Day-based queries ──
    if day:
        # Count queries
        if re.search(r'how many\s+(periods|subjects|classes|lectures)', q):
            return ("day_count", {"day": day})

        has_slot_kw = bool(re.search(r'time\s*slot|slot|period|timing', q))
        has_subj_kw = bool(re.search(r'subject|class|lecture|course', q))
        has_both_kw = bool(re.search(r'(along\s+with|with|and)', q))
        has_schedule_kw = bool(re.search(r'schedule|timetable for', q))

        if has_schedule_kw:
            return ("day_schedule", {"day": day})
        if has_slot_kw and (has_subj_kw or has_both_kw):
            return ("day_schedule", {"day": day})
        if has_slot_kw and not has_subj_kw:
            return ("day_slots", {"day": day})
        # Default for day queries: show subjects
        return ("day_subjects", {"day": day})

    # ── Faculty lookup ──
    if re.search(r'who\s+teaches|\bfaculty\b|teacher\s+(for|of)|taught\s+by', q):
        return ("faculty_lookup", {})

    # ── Metadata ──
    if re.search(r'\b(semester|section|year|department|issue\s*number|effective\s*date|w\.?e\.?f)\b', q):
        return ("metadata", {})

    # ── Almanac / events ──
    if re.search(r'\b(exams?|holidays?|internals?|assessments?|events?|almanac|fest|convocation)\b', q):
        return ("almanac", {})

    return (None, {})


def try_timetable_direct_answer(question, docs):
    """Attempt to answer structured timetable queries directly from
    retrieved chunks without calling the LLM.

    Returns the answer string if successful, or None to fall through to the LLM.
    """
    intent, params = classify_timetable_intent(question)
    if intent is None:
        return None

    print(f"[TIMETABLE] Intent: {intent} | Params: {params}")

    # ── Day-based intents ──
    if intent in ("day_subjects", "day_slots", "day_schedule", "day_count"):
        target_day = params["day"]
        day_doc = None
        for d in docs:
            if d.metadata.get("day", "").upper() == target_day:
                day_doc = d
                break
        if not day_doc:
            return None

        # Parse "- TIME: SUBJECT" lines from the chunk
        slot_lines = []
        for line in day_doc.page_content.split("\n"):
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                # e.g. "- 9:00-9:50 AM: SUBJECT"
                parts = line[2:].split(": ", 1)
                if len(parts) == 2:
                    time_range = parts[0].strip()
                    subjects_raw = parts[1].strip()
                    slot_lines.append((time_range, subjects_raw))

        if not slot_lines:
            return None

        if intent == "day_count":
            return f"There are **{len(slot_lines)}** periods on {target_day.title()}."

        result_lines = [f"**{target_day.title()} Schedule:**\n"]
        for time_range, subjects_raw in slot_lines:
            # Parallel block detection: multiple comma-separated entries
            entries = [e.strip() for e in subjects_raw.split(",")]

            if intent == "day_slots":
                result_lines.append(f"- {time_range}")
            elif intent == "day_subjects":
                if len(entries) > 1:
                    result_lines.append(f"- {time_range} [Parallel Block]:")
                    for entry in entries:
                        result_lines.append(f"  - {entry}")
                else:
                    result_lines.append(f"- {subjects_raw}")
            else:  # day_schedule — show both
                if len(entries) > 1:
                    result_lines.append(f"- {time_range} [Parallel Block]:")
                    for entry in entries:
                        result_lines.append(f"  - {entry}")
                else:
                    result_lines.append(f"- {time_range} — {subjects_raw}")

        return "\n".join(result_lines)

    # ── Faculty lookup ──
    if intent == "faculty_lookup":
        faculty_doc = None
        for d in docs:
            if d.metadata.get("doc_type") == "timetable_faculty":
                faculty_doc = d
                break
        if not faculty_doc:
            return None

        q_lower = question.lower()
        matches = []
        for line in faculty_doc.page_content.split("\n"):
            if line.startswith("- ") and "|" in line:
                line_lower = line.lower()
                # Check if any query keyword matches this faculty line
                query_words = [w for w in re.findall(r'\w{3,}', q_lower)
                               if w not in ('who', 'teaches', 'faculty', 'for', 'the', 'what')]
                if any(w in line_lower for w in query_words):
                    matches.append(line.strip())

        if matches:
            return "\n".join(matches)
        return None

    # ── Metadata ──
    if intent == "metadata":
        meta_doc = None
        for d in docs:
            if d.metadata.get("doc_type") == "timetable_metadata" and \
               d.metadata.get("metadata_section") == "header":
                meta_doc = d
                break
        if not meta_doc:
            return None

        # Look for parsed fields in the chunk content
        q_lower = question.lower()
        result_lines = []
        for line in meta_doc.page_content.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("- ") and ":" in line_stripped:
                field_name = line_stripped.split(":")[0].replace("- ", "").strip().lower()
                if any(kw in q_lower for kw in field_name.split("_")):
                    result_lines.append(line_stripped)

        if result_lines:
            return "\n".join(result_lines)
        return None

    # ── Almanac ──
    if intent == "almanac":
        almanac_doc = None
        for d in docs:
            if d.metadata.get("doc_type") == "timetable_almanac":
                almanac_doc = d
                break
        if not almanac_doc:
            return None

        q_lower = question.lower()
        matches = []
        for line in almanac_doc.page_content.split("\n"):
            if line.startswith("- "):
                line_lower = line.lower()
                query_words = [w for w in re.findall(r'\w{3,}', q_lower)
                               if w not in ('when', 'what', 'are', 'the', 'any', 'list', 'tell')]
                if any(w in line_lower for w in query_words):
                    matches.append(line.strip())

        if matches:
            return "**Academic Events:**\n" + "\n".join(matches)
        # If no keyword match, return full almanac
        return almanac_doc.page_content

    return None
