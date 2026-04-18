# Smart Campus Assistant — Document Vectorization Script
# Processes institutional PDFs and TXT files (regulations, timetable, code of conduct, etc.)
# and stores them in a ChromaDB vector database for RAG retrieval.
#
# Usage:  python vectorize_documents.py
#
# Place all PDF and/or TXT files in the ./data/ directory (including subfolders) before running.

import os
import re
import shutil
import traceback
from PyPDF2 import PdfReader
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Patterns that indicate corrupted/binary content in extracted text
BINARY_NOISE_PATTERNS = [
    r'Ljava[/\\]lang[/\\]',          # Java bytecode class references
    r'\.getInputStream\(\)',           # Java method signatures
    r'\.class\b',                      # Java .class references
    r'\\x[0-9a-fA-F]{2}',             # Hex escape sequences
    r'[\x00-\x08\x0e-\x1f\x7f-\x9f]', # Non-printable control characters
    r'endstream|endobj|xref',          # PDF internal markers
    r'stream\s*\n.*?endstream',        # PDF stream blocks
    r'/Type\s*/\w+',                   # PDF type declarations
    r'/Filter\s*/\w+',                 # PDF filter declarations
    r'%PDF-',                          # PDF magic bytes
]


def clean_extracted_text(text):
    """Clean raw extracted text to improve chunk quality.
    
    - Removes binary/bytecode artifacts (Java signatures, PDF internals)
    - Collapses excessive whitespace
    - Normalizes line breaks
    - Removes stray page numbers and headers
    """
    # Remove binary/bytecode noise patterns
    for pattern in BINARY_NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Remove lines that are mostly non-ASCII (likely binary garbage)
    cleaned_lines = []
    for line in text.split('\n'):
        if len(line) == 0:
            cleaned_lines.append(line)
            continue
        ascii_ratio = sum(1 for c in line if c.isascii() and (c.isprintable() or c in '\t\n\r')) / len(line)
        if ascii_ratio >= 0.85:  # Keep lines that are at least 85% clean ASCII
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    # Remove standalone page numbers (e.g., lines like "  12  " or "Page 5")
    text = re.sub(r'(?m)^\s*(Page\s*)?\d{1,3}\s*$', '', text)
    # Collapse multiple blank lines into at most two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse spaces (but keep newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def is_noisy_chunk(chunk_text, threshold=0.70):
    """Return True if a chunk is likely binary noise or bytecode garbage.
    
    Checks for:
    - High ratio of non-printable / non-ASCII characters
    - Known bytecode pattern matches
    - Very short chunks with no alphabetic content
    """
    if not chunk_text or len(chunk_text.strip()) < 10:
        return True
    
    # Check ASCII printable ratio
    printable_count = sum(1 for c in chunk_text if c.isascii() and (c.isprintable() or c in '\t\n\r'))
    ratio = printable_count / len(chunk_text)
    if ratio < threshold:
        return True
    
    # Check for known bytecode/binary patterns
    noise_indicators = [
        'Ljava/lang', 'Ljava\\lang', '.getInputStream',
        'endstream', 'endobj', '/Filter', '/Type /',
        '%PDF-', '\\x00', 'xref', 'startxref',
    ]
    noise_hits = sum(1 for pattern in noise_indicators if pattern in chunk_text)
    if noise_hits >= 2:
        return True
    
    # Check if chunk has almost no alphabetic words
    alpha_chars = sum(1 for c in chunk_text if c.isalpha())
    if alpha_chars / max(len(chunk_text), 1) < 0.30:
        return True
    
    return False


def is_binary_file(file_path):
    """Check if a file is binary by reading its first few bytes."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        # Check for common binary signatures
        binary_signatures = [b'%PDF-', b'\x89PNG', b'\xff\xd8\xff', b'PK\x03\x04', b'\x00\x00']
        return any(header.startswith(sig) for sig in binary_signatures)
    except Exception:
        return False


# ──────────────────────────────────────────────
# Day-name detection pattern (generalized, not hardcoded)
# ──────────────────────────────────────────────
_DAY_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}


def _is_day_name(cell_text):
    """Return True if the cell contains a recognizable day-of-week name."""
    if not cell_text:
        return False
    return cell_text.strip().lower() in _DAY_NAMES


def _is_empty_cell(cell):
    """Return True if a cell is empty / NULL / dash."""
    if cell is None:
        return True
    s = str(cell).strip()
    return s in ("", "None", "—", "-", "NULL")


def _find_schedule_col_range(header):
    """Dynamically detect the contiguous range of time-slot columns.
    
    Time-slot headers typically contain digits (e.g., '9:00-9:50 AM').
    Columns after a gap of non-time headers (like 'ROOM NUMBER') are excluded.
    Returns (start_col, end_col) — both inclusive.
    """
    time_cols = []
    for idx, h in enumerate(header):
        # Skip the first column (usually the day label)
        if idx == 0:
            continue
        if h and re.search(r'\d', h):  # Contains at least one digit → likely a time slot
            time_cols.append(idx)
    
    if not time_cols:
        # Fallback: treat all columns except the first as schedule columns
        return (1, len(header) - 1)
    
    # Return the contiguous range from first time-col to last time-col
    return (time_cols[0], time_cols[-1])


# ──────────────────────────────────────────────
# Timetable Region Extractors
# ──────────────────────────────────────────────

def _extract_header_metadata(page=None, source_filename="Unknown", raw_header_text=None):
    """Extract text OUTSIDE table bounding boxes (header area above the grid).

    Captures: institute name, department, semester, section, issue number,
    effective date — all of which sit above the main timetable table.
    Returns a single Document or None.
    """
    if raw_header_text is not None:
        header_text = raw_header_text
        sorted_lines = [l for l in header_text.split('\n') if l.strip()]
    else:
        words = page.extract_words()
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]

        # Filter words that do NOT fall inside any table bounding box
        outside_words = []
        for w in words:
            in_table = any(
                w['x0'] >= b[0] - 2 and w['top'] >= b[1] - 2 and
                w['x1'] <= b[2] + 2 and w['bottom'] <= b[3] + 2
                for b in table_bboxes
            )
            if not in_table:
                outside_words.append(w)

        if not outside_words:
            return None

        # Group words into lines by y-position (tolerance of 5 pts)
        lines_dict = {}
        for w in outside_words:
            y_key = round(w['top'], 0)
            matched = False
            for existing_y in list(lines_dict.keys()):
                if abs(existing_y - y_key) < 5:
                    lines_dict[existing_y].append(w)
                    matched = True
                    break
            if not matched:
                lines_dict[y_key] = [w]

        # Reconstruct lines sorted top-to-bottom, words sorted left-to-right
        sorted_lines = []
        for y in sorted(lines_dict.keys()):
            line_words = sorted(lines_dict[y], key=lambda w: w['x0'])
            line_text = " ".join(w['text'] for w in line_words)
            sorted_lines.append(line_text)

        header_text = "\n".join(sorted_lines)

    # ── Parse structured fields ──
    parsed = {}

    # ── PRIORITY SUBJECT DETECTION (TOP HEADING) ──
    lines = [l.strip() for l in header_text.split("\n") if l.strip()]

    for line in lines[:20]:
        clean = re.sub(r'[^A-Za-z\s]', '', line)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Fix broken words from PDF extraction
        clean = clean.replace("Comp Iler", "Compiler")
        clean = clean.replace("Data Base", "Database")
        clean = clean.replace("Oper ating", "Operating")

        if (
            len(clean) > 5 and
            len(clean.split()) >= 2 and
            (
                clean.isupper() or
                clean.istitle() or
                (len(clean.split()) <= 6 and len(clean.split()) >= 2)
            ) and
            not re.search(r"(GOKARAJU|INSTITUTE|ENGINEERING|TECHNOLOGY)", clean, re.IGNORECASE) and
            not re.search(r"^(UNIT|MODULE)\b", clean, re.IGNORECASE)
        ):
            if "subject" not in parsed:
                parsed["subject"] = clean.title()
                parsed["subject"] = re.sub(r'\s+', ' ', parsed["subject"]).strip()
            break

    if "subject" in parsed:
        if len(parsed["subject"].split()) < 2:
            parsed.pop("subject")

    for i, line in enumerate(lines):
        if "course code" in line.lower():
            for j in range(i-1, max(i-6, 0), -1):
                candidate = lines[j].strip()
                clean = re.sub(r'[^A-Za-z\s]', '', candidate)
                clean = re.sub(r'\s+', ' ', clean).strip()

                if (
                    len(clean) > 5 and
                    len(clean.split()) >= 2 and
                    "GOKARAJU" not in clean and
                    "INSTITUTE" not in clean
                ):
                    if "subject" not in parsed:
                        parsed["subject"] = clean.title()
                        parsed["subject"] = re.sub(r'\s+', ' ', parsed["subject"]).strip()
                    break
            break

    # Institute name — typically the first line
    if sorted_lines:
        parsed["institute"] = sorted_lines[0].strip()

    # Department — line containing "department"
    for line in sorted_lines:
        if "department" in line.lower():
            parsed["department"] = line.strip()
            break

    # Semester / Year / Section — line containing "semester" or "b.tech"
    for line in sorted_lines:
        ll = line.lower()
        if "semester" in ll or "b.tech" in ll or "section" in ll:
            parsed["semester_info"] = line.strip()
            sem_match = re.search(r'(\w+)\s+semester', ll)
            if sem_match:
                parsed["semester"] = sem_match.group(0).title()
            sec_match = re.search(r'-\s*(\w)\s*section', ll)
            if sec_match:
                parsed["section"] = sec_match.group(1).upper()
            year_match = re.search(r'(I{1,4}V?)\s+B\.?Tech', line, re.IGNORECASE)
            if year_match:
                parsed["year"] = year_match.group(1) + " B.Tech"
            break

    # Issue number
    issue_match = re.search(r'issue\s*:\s*(\d+)', header_text, re.IGNORECASE)
    if issue_match:
        parsed["issue_number"] = issue_match.group(1)

    # Effective date (w.e.f)
    wef_match = re.search(r'w\.?e\.?f\.?:?\s*([\d-]+)', header_text, re.IGNORECASE)
    if wef_match:
        parsed["effective_date"] = wef_match.group(1)

    if "subject" not in parsed:
        parsed["subject"] = "Unknown Subject"
        parsed["subject"] = re.sub(r'\s+', ' ', parsed["subject"]).strip()

    print("[DEBUG SUBJECT]", parsed.get("subject"))

    # ── Build chunk content ──
    content_lines = [
        f"Timetable Header Metadata (Source: {source_filename})",
        f"Institute, Department, Semester, Section, Issue, and Effective Date information.",
        "",
        f"Raw Header Text:",
        header_text,
        "",
    ]

    if parsed:
        content_lines.append("Parsed Fields:")
        for key, value in parsed.items():
            content_lines.append(f"  - {key}: {value}")

    doc = Document(
        page_content="\n".join(content_lines),
        metadata={
            "source": source_filename,
            "doc_type": "timetable_metadata",
            "metadata_section": "header",
            **{f"meta_{k}": v for k, v in parsed.items()}
        }
    )
    return doc, parsed


def _extract_right_panel(tables, source_filename, header_meta):
    """Extract the right-side officials panel (HOD, Year Incharge, Section Coordinator).

    This is typically the smallest table on the page, containing role → name rows.
    Returns a single Document or None.
    """
    if not tables or len(tables) < 2:
        return None

    # The officials panel is the table with the fewest rows
    panel_table = min(tables, key=lambda t: len(t))

    if len(panel_table) < 1:
        return None

    content_lines = [
        f"Timetable Officials / Authority Panel (Source: {source_filename})",
        "Head of Department (HOD), Year Incharge, and Section Coordinator details:",
        "",
    ]

    officials = {}
    for row in panel_table:
        cells = [str(c).strip() if c else "" for c in row]
        non_empty = [c for c in cells if c]
        if len(non_empty) >= 2:
            role = non_empty[0]
            name = non_empty[1]
            officials[role] = name
            content_lines.append(f"- {role}: {name}")

    if not officials:
        return None

    doc = Document(
        page_content="\n".join(content_lines),
        metadata={
            "source": source_filename,
            "doc_type": "timetable_metadata",
            "metadata_section": "officials",
            **header_meta
        }
    )
    return doc


def _extract_subject_faculty_table(main_table, source_filename, header_meta):
    """Extract the subject/faculty mapping table from the lower portion of the main grid.

    Produces a chunk listing every subject code, name, abbreviation, and faculty name.
    Returns a single Document or None.
    """
    # Locate the header row containing "Subject Code"
    subject_header_idx = None
    for i, row in enumerate(main_table):
        cells = [str(c).strip().lower() if c else "" for c in row]
        if cells[0] == "subject code" or any("subject code" in c for c in cells):
            subject_header_idx = i
            break

    if subject_header_idx is None:
        return None

    # === DYNAMIC COLUMN DETECTION (HEADER SCANS) ===
    header_row = [str(c).strip().lower() if c else "" for c in main_table[subject_header_idx]]
    prev_row = [str(c).strip().lower() if c else "" for c in main_table[subject_header_idx - 1]] if subject_header_idx > 0 else []
    
    # SAFETY CONSTRAINT: Only use prev_row if it looks like a header row
    prev_row_valid = any(kw in str(c) for c in prev_row for kw in ["subject", "faculty", "name", "code"])
    if not prev_row_valid:
        prev_row = []

    code_idx, name_idx, faculty_short_idx, faculty_name_idx = -1, -1, -1, -1

    for idx in range(len(header_row)):
        cell_text = header_row[idx]
        prev_text = prev_row[idx] if idx < len(prev_row) else ""
        combined = (cell_text + " " + prev_text).strip()
        
        if "subject code" in combined:
            code_idx = idx
        elif "subject name" in combined:
            name_idx = idx
        elif "faculty name" in combined:
            faculty_name_idx = idx
        elif "faculty" in combined and "faculty name" not in combined:
            faculty_short_idx = idx
            
    # Safe fallbacks if extraction shifts slightly
    code_idx = code_idx if code_idx != -1 else 0
    name_idx = name_idx if name_idx != -1 else 1
    faculty_short_idx = faculty_short_idx if faculty_short_idx != -1 else 5
    faculty_name_idx = faculty_name_idx if faculty_name_idx != -1 else 6
    # ===============================================

    content_lines = [
        f"Subject and Faculty Table (Source: {source_filename})",
        "List of subjects, subject codes, abbreviations, and faculty members:",
        "",
    ]

    current_entry = None
    entries = []

    for row in main_table[subject_header_idx + 1:]:
        cells = [str(c).strip() if c else "" for c in row]

        subject_code = cells[code_idx].strip() if len(cells) > code_idx else ""
        subject_name = cells[name_idx].strip() if len(cells) > name_idx else ""
        faculty_short = cells[faculty_short_idx] if len(cells) > faculty_short_idx else ""
        faculty_name = cells[faculty_name_idx] if len(cells) > faculty_name_idx else ""

        # Clean embedded newlines in faculty names
        faculty_name = faculty_name.replace("\n", " ").strip()

        if subject_code or subject_name:
            if not subject_code or not subject_name:
                continue
            # New subject entry
            current_entry = {
                "code": subject_code,
                "name": subject_name,
                "faculty_short": faculty_short,
                "faculty_name": faculty_name,
            }
            entries.append(current_entry)
        elif (faculty_short or faculty_name) and current_entry:
            # Continuation row — additional faculty for the same subject
            if faculty_short and current_entry["faculty_short"]:
                current_entry["faculty_short"] += " / " + faculty_short
            elif faculty_short:
                current_entry["faculty_short"] = faculty_short
            if faculty_name and current_entry["faculty_name"]:
                current_entry["faculty_name"] += " | " + faculty_name
            elif faculty_name:
                current_entry["faculty_name"] = faculty_name

    # Format entries into semantic text
    for entry in entries:
        parts = []
        if entry["code"]:
            parts.append(f"Code: {entry['code']}")
        if entry["name"]:
            parts.append(f"Subject: {entry['name']}")
        if entry["faculty_short"]:
            parts.append(f"Abbreviation: {entry['faculty_short']}")
        if entry["faculty_name"]:
            parts.append(f"Faculty: {entry['faculty_name']}")
        if parts:
            content_lines.append("- " + " | ".join(parts))

    if not entries:
        return None

    doc = Document(
        page_content="\n".join(content_lines),
        metadata={
            "source": source_filename,
            "doc_type": "timetable_faculty",
            **header_meta
        }
    )
    return doc


def _extract_almanac_table(main_table, source_filename, header_meta):
    """Extract the almanac / academic events calendar from the right side of the main table.

    Returns a single Document or None.
    """
    # Locate the header row containing "EVENT"
    almanac_header_idx = None
    for i, row in enumerate(main_table):
        cells = [str(c).strip().lower() if c else "" for c in row]
        if any(c == "event" for c in cells):
            almanac_header_idx = i
            break

    if almanac_header_idx is None:
        # Fallback: look for "Almanac" section header and skip one extra row
        for i, row in enumerate(main_table):
            cells = [str(c).strip().lower() if c else "" for c in row]
            if any("almanac" in c for c in cells):
                almanac_header_idx = i + 1  # section header → column header is next
                break

    if almanac_header_idx is None:
        return None

    # === DYNAMIC COLUMN DETECTION for almanac header ===
    almanac_header = [str(c).strip().lower() if c else "" for c in main_table[almanac_header_idx]]
    event_col, period_col, duration_col = -1, -1, -1
    for idx, cell in enumerate(almanac_header):
        if "event" in cell:
            event_col = idx
        elif "period" in cell or "date" in cell:
            period_col = idx
        elif "duration" in cell or "days" in cell:
            duration_col = idx

    # If we couldn't find the event column, this isn't a valid almanac table
    if event_col == -1:
        return None

    content_lines = [
        f"Academic Almanac / Events Calendar (Source: {source_filename})",
        "List of academic events, dates, and durations for the semester:",
        "",
    ]

    events = []
    for row in main_table[almanac_header_idx + 1:]:
        cells = [str(c).strip() if c else "" for c in row]

        event_text = cells[event_col] if len(cells) > event_col else ""
        period_text = cells[period_col] if period_col != -1 and len(cells) > period_col else ""
        duration_text = cells[duration_col] if duration_col != -1 and len(cells) > duration_col else ""

        if event_text:
            event_clean = event_text.replace("\n", " ").strip()
            line = f"- {event_clean}"
            if period_text:
                line += f" | Period: {period_text}"
            if duration_text:
                line += f" | Duration: {duration_text}"
            events.append(line)
            content_lines.append(line)

    if not events:
        return None

    doc = Document(
        page_content="\n".join(content_lines),
        metadata={
            "source": source_filename,
            "doc_type": "timetable_almanac",
            **header_meta
        }
    )
    return doc


def _extract_day_schedule_chunks(main_table, source_filename, total_pages, header_meta):
    """Extract day-wise schedule chunks (Monday–Saturday) from the main timetable grid.

    Handles multi-row merged cells (e.g., lab sessions where the day label
    spans two rows and the first sub-row appears before the day name).
    Returns a list of Documents (one per day).
    """
    if len(main_table) < 2:
        return []

    header = [str(c).strip() if c else "" for c in main_table[0]]
    start_col, end_col = _find_schedule_col_range(header)
    time_slots = header[start_col:end_col + 1]

    # ── Collect only schedule rows (stop at empty separator or section headers) ──
    schedule_rows = []
    for row in main_table[1:]:
        cells = [str(c).strip() if c else "" for c in row]
        # Empty row = schedule section ended
        if not any(cells):
            break
        # Section header row = schedule section ended
        first_lower = cells[0].lower() if cells[0] else ""
        if first_lower in ("subjects", "subject code"):
            break
        if any("subject code" in c.lower() for c in cells if c):
            break
        schedule_rows.append(cells)

    if not schedule_rows:
        return []

    # ── Pre-scan: find the first day name (needed for orphan pre-rows) ──
    first_day = None
    for cells in schedule_rows:
        if _is_day_name(cells[0]):
            first_day = cells[0].strip().upper()
            break

    # ── Group rows by day — handles merged-cell continuation rows ──
    current_day = None
    current_slots = None
    day_groups = []  # list of (day_name, slot_lists)

    # Filter set for non-subject values in schedule cells
    SKIP_VALUES = {"room number", "break", "lunch", "theory", "tutorial"}

    for cells in schedule_rows:
        first_cell = cells[0].strip().lower() if cells[0] else ""

        if _is_day_name(first_cell):
            new_day = first_cell.upper()
            # Save previous day if switching to a different day
            if current_day and current_day != new_day and current_slots:
                day_groups.append((current_day, current_slots))
                current_slots = None
            # Initialize slots for a new day (or keep existing for same-day merged row)
            if current_day != new_day:
                current_day = new_day
                current_slots = [[] for _ in range(len(time_slots))]
        else:
            # Orphan row (merged-cell continuation or pre-day-label row)
            if current_day is None and first_day:
                current_day = first_day
                current_slots = [[] for _ in range(len(time_slots))]

        # ── Collect subject values from schedule columns ──
        if current_day is not None and current_slots is not None:
            for i, col_idx in enumerate(range(start_col, end_col + 1)):
                if col_idx >= len(cells):
                    continue
                value = cells[col_idx]
                if (value
                        and not _is_empty_cell(value)
                        and value.lower() not in SKIP_VALUES):
                    current_slots[i].append(value)

    # Save the last day group
    if current_day and current_slots:
        day_groups.append((current_day, current_slots))

    # ── Build one Document per day ──
    day_documents = []
    for day_name, slot_data in day_groups:
        lines = [
            f"{day_name} Schedule (Source: {source_filename})",
            "Time Slots:",
        ]
        subjects_summary = []

        for idx, entries in enumerate(slot_data):
            if not entries:
                continue
            time_label = (
                time_slots[idx] if idx < len(time_slots) else f"Slot {idx + 1}"
            )
            clean_entries = list(dict.fromkeys(entries))  # deduplicate, preserve order
            subject_line = ", ".join(clean_entries)
            lines.append(f"- {time_label}: {subject_line}")
            subjects_summary.extend(clean_entries)

        unique_subjects = list(dict.fromkeys(subjects_summary))

        if unique_subjects:
            lines.append(
                f"\nSubjects on {day_name}: " + ", ".join(unique_subjects)
            )

        doc = Document(
            page_content="\n".join(lines),
            metadata={
                "source": source_filename,
                "doc_type": "timetable_day",
                "day": day_name,
                "total_pages": total_pages,
                **header_meta
            }
        )
        day_documents.append(doc)
        print(f"    -> {day_name}: {len(unique_subjects)} subjects extracted")

    return day_documents


def _extract_full_page_backup(page, page_number, source_filename, header_meta):
    """Extract full visible text from a page using pdfplumber.extract_text().

    Stored as a fallback chunk so no visible text region is ever missed.
    Returns a single Document or None.
    """
    text = page.extract_text()
    if not text or not text.strip():
        return None

    doc = Document(
        page_content=(
            f"Full Page Text Backup (Source: {source_filename}, Page {page_number})\n"
            f"This contains every visible text region on page {page_number}.\n\n"
            f"{text.strip()}"
        ),
        metadata={
            "source": source_filename,
            "doc_type": "timetable_full_backup",
            "page": page_number,
            **header_meta
        }
    )
    return doc


# ──────────────────────────────────────────────
# Master Timetable Extractor
# ──────────────────────────────────────────────

def extract_timetable_all_chunks(file_path, source_filename):
    """Extract ALL timetable regions into semantically typed chunks.

    Regions extracted:
      1. Header metadata  (institute, department, semester, section, issue, date)
      2. Officials panel   (HOD, Year Incharge, Section Coordinator)
      3. Day-wise schedule (one chunk per day, Monday–Saturday)
      4. Subject / Faculty table
      5. Almanac / Events calendar
      6. Full-page text backup (fallback)

    Returns a list of Documents.
    """
    all_chunks = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            # ── 1. Header metadata (text outside table bboxes) ──
            header_meta = {}
            header_res = _extract_header_metadata(page, source_filename)
            if header_res:
                header_doc, header_meta = header_res
                all_chunks.append(header_doc)
                print(f"    -> Header metadata extracted")

            # ── 2. Officials / right-side panel ──
            if tables and len(tables) >= 2:
                panel_doc = _extract_right_panel(tables, source_filename, header_meta)
                if panel_doc:
                    all_chunks.append(panel_doc)
                    print(f"    -> Officials panel extracted")

            # ── 3–5. Main table regions ──
            if tables:
                # Select the main (largest) table
                main_table = max(
                    tables,
                    key=lambda t: (len(t), max(len(r) for r in t if r))
                )

                # 3. Day-wise schedule chunks
                day_docs = _extract_day_schedule_chunks(
                    main_table, source_filename, len(pdf.pages), header_meta
                )
                all_chunks.extend(day_docs)

                # 4. Subject / Faculty table
                faculty_doc = _extract_subject_faculty_table(
                    main_table, source_filename, header_meta
                )
                if faculty_doc:
                    all_chunks.append(faculty_doc)
                    print(f"    -> Subject/Faculty table extracted")

                # 5. Almanac / Events calendar
                almanac_doc = _extract_almanac_table(
                    main_table, source_filename, header_meta
                )
                if almanac_doc:
                    all_chunks.append(almanac_doc)
                    print(f"    -> Almanac/Events table extracted")

            # ── 6. Full-page text backup ──
            backup_doc = _extract_full_page_backup(
                page, page_num, source_filename, header_meta
            )
            if backup_doc:
                all_chunks.append(backup_doc)
                print(f"    -> Full-page text backup extracted")

    return all_chunks


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Semantic Syllabus Extractor
# ──────────────────────────────────────────────

def extract_syllabus_chunks(full_text, source_filename):
    """Semantically chunk syllabus PDFs by Semester -> Subject -> Unit."""
    chunks = []
    lines = full_text.split('\n')
    
    inferred_semester = "Unknown"
    inferred_subject = "Unknown"
    
    sem_pattern = re.compile(r'((?:I|II|III|IV)\s*Year\s*(?:I|II)\s*Semester|(?:I|II|III|IV|V|VI|VII|VIII)\s*Semester)', re.IGNORECASE)
    course_code_pattern = re.compile(r'(?:Course Code|Subject Code)\s*[:\-]?\s*([A-Z0-9]+)', re.IGNORECASE)
    subj_pattern = re.compile(r'(?:Subject|Course Name|Course Title|Subject Name)\s*[:\-]\s*(.+)', re.IGNORECASE)
    unit_pattern = re.compile(r'^(UNIT|MODULE)\s*[-|:]?\s*([IVXLC\d]+)', re.IGNORECASE)
    topic_pattern = re.compile(r'^([A-Z][a-zA-Z\s\&\-]{2,30})\s*[:\-]?$')

    # 1. Filename heuristic
    import os
    base_name = os.path.basename(source_filename).replace('_', ' ').replace('-', ' ')
    
    sem_file_match = sem_pattern.search(base_name)
    if sem_file_match:
        inferred_semester = sem_file_match.group(1).title()
        
    clean_name = re.sub(r'(?i)\b(syllabus|curriculum|b\.?tech|m\.?tech|course|pdf)\b', '', base_name)
    clean_name = re.sub(r'((?:I|II|III|IV)\s*Year\s*(?:I|II)\s*Semester|(?:I|II|III|IV|V|VI|VII|VIII)\s*Semester)', '', clean_name, flags=re.IGNORECASE)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    if clean_name and len(clean_name) > 3:
        inferred_subject = clean_name.title()

    # 2. First page headings global pre-scan
    if inferred_semester == "Unknown":
        for line in lines[:200]:
            sem_match = sem_pattern.search(line)
            if sem_match:
                inferred_semester = sem_match.group(1).title()
                break

    if inferred_subject == "Unknown":
        for line in lines[:100]:
            l = line.strip()
            if l.isupper() and 5 < len(l) < 60:
                if not re.search(r'(SYLLABUS|ENGINEERING|INSTITUTE|DEPARTMENT|UNIVERSITY|SEMESTER|YEAR|COURSE STRUCTURE|OBJECTIVES|OUTCOMES)', l, re.IGNORECASE):
                    inferred_subject = l.title()
                    break
                    
    current_semester = inferred_semester
    current_subject = inferred_subject
    current_course_code = "Unknown"
    current_unit = "General"
    current_content = []
    
    def save_chunk():
        if current_content:
            text = "\n".join(current_content).strip()
            if len(text) > 15:
                # Prepend the context so LLM knows what this chunk is about internally
                context_header = f"Semester: {current_semester} | Subject: {current_subject} | Code: {current_course_code} | {current_unit}\n\n"
                
                stopwords = {
                    "this", "that", "with", "from", "have", "will", "using", "based", "given",
                    "into", "their", "there", "which", "where", "when", "while",
                    "system", "method", "approach", "introduction", "analysis"
                }

                words = [
                    w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                    if w not in stopwords
                ]

                keywords = " ".join(dict.fromkeys(words[:30]))

                if len(text) > 800:
                    sub_parts = re.split(r'\.\s+|\n', text)
                    temp = ""
                    for part in sub_parts:
                        if len(temp) + len(part) < 600:
                            temp += " " + part
                        else:
                            doc = Document(
                                page_content=context_header + f"Keywords: {keywords}\n\n" + temp.strip(),
                                metadata={
                                    "source": source_filename,
                                    "doc_type": "syllabus",
                                    "semester": current_semester,
                                    "subject": current_subject,
                                    "course_code": current_course_code,
                                    "unit": current_unit
                                }
                            )
                            chunks.append(doc)
                            temp = part
                    if temp:
                        doc = Document(
                            page_content=context_header + f"Keywords: {keywords}\n\n" + temp.strip(),
                            metadata={
                                "source": source_filename,
                                "doc_type": "syllabus",
                                "semester": current_semester,
                                "subject": current_subject,
                                "course_code": current_course_code,
                                "unit": current_unit
                            }
                        )
                        chunks.append(doc)
                    current_content.clear()
                    return

                doc = Document(
                    page_content=context_header + f"Keywords: {keywords}\n\n" + text,
                    metadata={
                        "source": source_filename,
                        "doc_type": "syllabus",
                        "semester": current_semester,
                        "subject": current_subject,
                        "course_code": current_course_code,
                        "unit": current_unit
                    }
                )
                chunks.append(doc)
            current_content.clear()
            
    pending_subject = False
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            current_content.append(line)
            continue
            
        sem_match = sem_pattern.search(line_clean)
        if sem_match:
            current_semester = sem_match.group(1).strip()
            
        code_match = course_code_pattern.search(line_clean)
        if code_match:
            current_course_code = code_match.group(1).strip()
            
        subj_match = subj_pattern.search(line_clean)
        if subj_match:
            save_chunk()
            current_subject = subj_match.group(1).strip()
            current_unit = "General"
            pending_subject = False
        elif line_clean.upper() in ["SUBJECT", "COURSE TITLE", "COURSE NAME"]:
            save_chunk()
            pending_subject = True
            current_content.append(line)
            continue
        elif pending_subject and len(line_clean) > 3 and line_clean.isupper():
            current_subject = line_clean
            current_unit = "General"
            pending_subject = False
            
        unit_match = unit_pattern.search(line_clean)
        if unit_match:
            save_chunk()
            current_unit = f"{unit_match.group(1).upper()}-{unit_match.group(2).upper()}"
            
        # Topic heading split if chunk getting large
        topic_match = topic_pattern.search(line_clean)
        if topic_match and len("\n".join(current_content)) > 1000:
            save_chunk()
        
        current_content.append(line)
        
    save_chunk()
    return chunks

# ──────────────────────────────────────────────
# Semantic Regulation Extractor
# ──────────────────────────────────────────────

def extract_regulation_chunks(full_text, source_filename, doc_type):
    """Semantically chunk regulation PDFs by section headings."""
    chunks = []
    lines = full_text.split('\n')
    
    current_section = "General"
    current_version = "R22" # default
    current_content = []
    
    ver_match = re.search(r'(R\d{2})', full_text)
    if ver_match:
        current_version = ver_match.group(1)
        
    section_keywords = ["ATTENDANCE", "CREDIT", "PROMOTION", "BACKLOG", "EXAMINATION", "DETENTION", "EVALUATION", "AWARD OF DEGREE", "ACADEMIC", "CONDUCT"]
    
    def is_section_header(text):
        text_upper = text.upper()
        if len(text) > 80:
            return False
        return any(kw in text_upper for kw in section_keywords)

    def save_chunk():
        if current_content:
            text = "\n".join(current_content).strip()
            if len(text) > 15:
                context_header = f"Regulation Version: {current_version}\nSection: {current_section}\n\n"
                doc = Document(
                    page_content=context_header + text,
                    metadata={
                        "source": source_filename,
                        "doc_type": doc_type,
                        "regulation_version": current_version,
                        "section": current_section
                    }
                )
                chunks.append(doc)
            current_content.clear()
            
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            current_content.append(line)
            continue
            
        match = re.match(r'^(\d+\.\d*)\s+(.*)', line_clean)
        if match:
            heading_text = match.group(2)
            if is_section_header(heading_text):
                save_chunk()
                current_section = heading_text.title()
                current_content.append(line)
                continue
                
        if len(line_clean) < 50 and line_clean.isupper() and is_section_header(line_clean):
            save_chunk()
            current_section = line_clean.title()
            
        current_content.append(line)
        
    save_chunk()
    return chunks

# ──────────────────────────────────────────────
# Document Classification (path-aware)
# ──────────────────────────────────────────────

def classify_document(relative_path):
    """Classify document type based on filename AND parent folder path.
    Final deterministic classification support for 5 categories only.
    """
    path_lower = relative_path.lower().replace("\\", "/")
    if "timetable" in path_lower or "schedule" in path_lower:
        return "timetable"
    elif "regulation" in path_lower or "conduct" in path_lower:
        return "regulations"
    elif "syllabus" in path_lower or "curriculum" in path_lower:
        return "syllabus"
    elif "notice" in path_lower:
        return "notices"
    elif "placement" in path_lower:
        return "placements"
    else:
        return None


def normalize_subject_name(raw_name: str):
    """
    Safely repair broken intra-word spacing without damaging valid subject names.
    Uses explicit fixes + dynamic dictionary validation.
    """
    import re

    STOPWORDS = {
        "AND", "OF", "FOR", "IN", "ON", "AT", "BY", "THE", "WITH",
        "BIG", "WEB", "FULL", "DATA", "JAVA", "CLOUD", "STACK"
    }

    KNOWN_FIXES = {
        "MA NAGEMENT": "MANAGEMENT",
        "MAT HEMATICS": "MATHEMATICS",
        "COMP ILER": "COMPILER",
        "PROGRAMMI NG": "PROGRAMMING",
        "OPER ATING": "OPERATING",
        "DATA BASE": "DATABASE",
        "AUTO MATA": "AUTOMATA",
    }

    VALID_WORDS = {
        "MANAGEMENT", "MATHEMATICS", "COMPILER", "PROGRAMMING", "OPERATING",
        "DATABASE", "SYSTEMS", "ENGINEERING", "SCIENCE", "TECHNOLOGY",
        "DESIGN", "ANALYSIS", "ALGORITHMS", "STRUCTURES", "NETWORKS",
        "SECURITY", "INTELLIGENCE", "LEARNING", "CLOUD", "COMPUTING",
        "DEVELOPMENT", "APPLICATION", "MOBILE", "WEB", "BIG", "DATA",
        "WAREHOUSING", "MINING", "PRINCIPLES", "LANGUAGES", "DISCRETE",
        "PROBABILITY", "STATISTICS", "DIGITAL", "LOGIC", "ORGANIZATION",
        "ARCHITECTURE", "COMMUNICATION", "ECONOMICS", "ACCOUNTING",
        "ELECTRICAL", "CHEMISTRY", "PHYSICS", "BIOLOGY", "LINEAR",
        "ALGEBRA", "CALCULUS", "DIFFERENTIAL", "EQUATIONS", "VECTOR",
        "FUNCTION", "APPROXIMATION", "AUTOMATA", "THEORY", "COMPUTER",
        "GRAPHICS", "CYBER", "FULL", "STACK", "UNIX", "OBJECT",
        "ORIENTED", "ARTIFICIAL", "NEURAL", "NETWORK", "MACHINE",
        "DEEP", "REINFORCEMENT", "NATURAL", "LANGUAGE", "PROCESSING"
    }

    def is_valid_merged_word(w: str) -> bool:
        return w.upper() in VALID_WORDS

    fixed = raw_name.strip()

    # Step 1: explicit known fixes
    for wrong, correct in KNOWN_FIXES.items():
        fixed = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, fixed, flags=re.IGNORECASE)

    # Step 2: dynamic merging with dictionary validation
    words = fixed.split()
    repaired = []
    i = 0
    while i < len(words):
        if i < len(words) - 1:
            first = words[i]
            second = words[i + 1]
            candidate = first + second

            if (
                first.upper() not in STOPWORDS
                and first.isupper()
                and second.isupper()
                and 2 <= len(first) <= 8
                and 2 <= len(second) <= 8
                and (len(first) == 1 or is_valid_merged_word(candidate))
            ):
                repaired.append(candidate)
                i += 2
                continue
        repaired.append(words[i])
        i += 1

    fixed = " ".join(repaired)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    return fixed


def split_syllabus_into_subjects(full_text, source_filename):
    """
    Parse a multi-subject syllabus PDF using "Course Code:" as block anchors.
    """
    docs = []
    
    # Remove page markers
    clean_text = re.sub(r'--- Page \d+ ---', '', full_text)
    
    semester = "Unknown"
    sem_match = re.search(r'sem(\d+)', source_filename, re.IGNORECASE)
    if sem_match:
        semester = f"Semester {sem_match.group(1)}"
    
    # Fallback mapping: course code -> subject name
    SUBJECT_MAP = {
        "GR22A1005": "ENGINEERING CHEMISTRY",
        "GR22A1008": "FUNDAMENTALS OF ELECTRICAL ENGINEERING",
        "GR22A3052": "SOFTWARE ENGINEERING",
        "GR22A2069": "DATABASE MANAGEMENT SYSTEMS",
        "GR22A2075": "DISCRETE MATHEMATICS",
        "GR22A3047": "PRINCIPLES OF PROGRAMMING LANGUAGES",
        "GR22A3115": "AUTOMATA AND COMPILER DESIGN",
        # Add more as needed from extraction logs
    }
    
    # Find all course code positions
    code_pattern = re.compile(r'Course\s*Code\s*:\s*([A-Z0-9]+)', re.IGNORECASE)
    matches = list(code_pattern.finditer(clean_text))
    
    if not matches:
        print(f"    WARNING: No course codes found in {source_filename}")
        return docs
    
    for i, match in enumerate(matches):
        course_code = match.group(1).strip()
        start_pos = match.start()
        
        # Determine block start: either previous match end or beginning of text
        if i > 0:
            prev_end = matches[i-1].end()
            block_text = clean_text[prev_end:start_pos] + match.group(0) + clean_text[start_pos:matches[i+1].start() if i+1 < len(matches) else len(clean_text)]
        else:
            # First subject: include everything before first course code as potential subject name
            block_text = clean_text[:start_pos] + match.group(0) + clean_text[start_pos:matches[i+1].start() if i+1 < len(matches) else len(clean_text)]
        
        # Extract subject name: look backwards from course code for all-caps line
        # Increased window to 800 for better discovery
        before_code = clean_text[max(0, start_pos-800):start_pos]
        lines_before = before_code.split('\n')
        subject_name = None
        for line in reversed(lines_before):
            line = line.strip()
            # Regex allows digits and hyphens. Excludes admin headers and course markers.
            if re.match(r'^[A-Z][A-Z0-9\s&\-]{4,}$', line) and not re.search(r'GOKARAJU|INSTITUTE|ENGINEERING|TECHNOLOGY|DEPARTMENT|UNIT|COURSE|CODE|L/T/P/C|SEMESTER|YEAR', line, re.IGNORECASE):
                subject_name = line
                break
        
        if not subject_name:
            # Fallback: look forward up to 10 lines
            after_code = clean_text[match.end():match.end()+500]
            lines_after = after_code.split('\n')
            for line in lines_after:
                line = line.strip()
                if re.match(r'^[A-Z][A-Z0-9\s&\-]{4,}$', line) and not re.search(r'GOKARAJU|INSTITUTE|UNIT|COURSE|CODE|L/T/P/C|PREREQUISITES|OUTCOMES|SEMESTER|YEAR', line, re.IGNORECASE):
                    subject_name = line
                    break
        
        # Post-extraction fallback using hardcoded mapping
        if not subject_name or subject_name == "Unknown Subject":
            subject_name = SUBJECT_MAP.get(course_code, "Unknown Subject")
            if subject_name != "Unknown Subject":
                print(f"    [DEBUG] Used fallback mapping for {course_code}: {subject_name}")
        
        # Subject Name Normalization (Repair PDF spacing artifacts)
        original_name = subject_name
        subject_name = normalize_subject_name(subject_name)
        if original_name != subject_name and subject_name != "Unknown Subject":
            print(f"    [DEBUG] Subject name normalized: '{original_name}' -> '{subject_name}'")
        
        # Get content: from this course code to next course code (or end)
        content_end = matches[i+1].start() if i+1 < len(matches) else len(clean_text)
        content = clean_text[start_pos:content_end]
        
        content_clean = clean_extracted_text(content)
        if len(content_clean) < 200:
            continue
        
        # Hybrid Chunking Implementation
        first_unit_anchor = re.compile(r'(UNIT\s*[-:]?\s*I\b|UNIT\s*[-:]?\s*1\b)', re.IGNORECASE)
        unit_header_pattern = re.compile(r'(UNIT\s*[-:]?\s*[IVXLCDM]+|UNIT\s*[-:]?\s*\d+)', re.IGNORECASE)
        
        match = first_unit_anchor.search(content_clean)
        
        if match:
            split_pos = match.start()
            pre_unit_content = content_clean[:split_pos].strip()
            unit_content = content_clean[split_pos:].strip()
            
            meta_count = 0
            unit_count = 0
            
            # 1. Create SUBJECT_META chunk
            if len(pre_unit_content) > 50:
                docs.append(Document(
                    page_content=pre_unit_content,
                    metadata={
                        "source": source_filename,
                        "doc_type": "syllabus",
                        "subject": subject_name,
                        "course_code": course_code,
                        "semester": semester,
                        "chunk_type": "subject_meta"
                    }
                ))
                meta_count += 1
            
            # 2. Create UNIT chunks
            parts = unit_header_pattern.split(unit_content)
            # parts will be ['', 'UNIT I', ' body...', 'UNIT II', ' body...', ...]
            for j in range(1, len(parts), 2):
                header = parts[j]
                body = parts[j+1] if j+1 < len(parts) else ""
                combined = (header + body).strip()
                if combined:
                    docs.append(Document(
                        page_content=combined,
                        metadata={
                            "source": source_filename,
                            "doc_type": "syllabus",
                            "subject": subject_name,
                            "course_code": course_code,
                            "semester": semester,
                            "chunk_type": "unit",
                            "unit": header.strip().upper()
                        }
                    ))
                    unit_count += 1
            print(f"    -> Created {meta_count} meta + {unit_count} unit chunks for {subject_name}")
            
        else:
            # Fallback (Step 4: create single document if no units found)
            doc = Document(
                page_content=content_clean,
                metadata={
                    "source": source_filename,
                    "doc_type": "syllabus",
                    "subject": subject_name,
                    "course_code": course_code,
                    "semester": semester
                }
            )
            docs.append(doc)
            print(f"    -> Extracted subject: {subject_name} ({course_code})")
    
    return docs


# ──────────────────────────────────────────────
# Document Loaders (recursive via os.walk)
# ──────────────────────────────────────────────

def load_pdf_documents(directory):
    """Load and extract text from PDF files, recursively scanning all subfolders.

    - Timetable/schedule PDFs → multi-region extraction via pdfplumber
    - All other PDFs → standard text extraction via PyPDF2

    Returns: (regular_documents, timetable_chunks)
      - regular_documents: need generic text splitting
      - timetable_chunks: already granular, skip the text splitter
    """
    regular_documents = []
    timetable_chunks = []

    # Recursively discover all PDF files
    pdf_files = []  # list of (full_path, relative_path)
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith('.pdf'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                pdf_files.append((full_path, rel_path))

    if not pdf_files:
        print("No PDF files found in the data directory (including subfolders).")
        return regular_documents, timetable_chunks

    print(f"  Found {len(pdf_files)} PDF file(s):")
    for _, rp in pdf_files:
        print(f"    - {rp}")

    for full_path, rel_path in pdf_files:
        try:
            print(f"\nProcessing {rel_path}...")
            doc_type = classify_document(rel_path)
            
            if doc_type is None:
                print(f"  [SKIP] Skipping unsupported file: {rel_path}")
                continue

            # Route timetable PDFs through the multi-region extractor
            if doc_type == "timetable":
                print(f"  Using pdfplumber (multi-region timetable extraction)...")
                chunks = extract_timetable_all_chunks(full_path, rel_path)
                timetable_chunks.extend(chunks)
                print(f"  [OK] Extracted {len(chunks)} chunks from {rel_path}")
            elif doc_type == "syllabus":
                print(f"  Using multi-subject syllabus parsing...")
                reader = PdfReader(full_path)
                full_text = ""
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n--- Page {page_num} ---\n\n"
                        full_text += page_text
                full_text = clean_extracted_text(full_text)
                if not full_text.strip() or len(full_text.strip()) < 200:
                    print(f"  ❌ SKIPPED (empty or scanned PDF): {rel_path}")
                else:
                    subject_docs = split_syllabus_into_subjects(full_text, rel_path)
                    timetable_chunks.extend(subject_docs)
                    print(f"  [OK] Extracted {len(subject_docs)} subject documents from {rel_path}")
            elif doc_type == "regulations":
                print(f"  Using semantic chunking for {doc_type}...")
                reader = PdfReader(full_path)
                full_text = ""
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n--- Page {page_num} ---\n\n"
                        full_text += page_text
                full_text = clean_extracted_text(full_text)
                if not full_text.strip() or len(full_text.strip()) < 200:
                    print(f"  ❌ SKIPPED (scanned or empty PDF): {rel_path}")
                    continue

                chunks = extract_regulation_chunks(full_text, rel_path, doc_type)
                timetable_chunks.extend(chunks)
                print(f"  [OK] Extracted {len(chunks)} semantic chunks from {rel_path}")
            else:
                # Standard text extraction with PyPDF2
                reader = PdfReader(full_path)
                full_text = ""

                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n--- Page {page_num} ---\n\n"
                        full_text += page_text

                total_pages = len(reader.pages)

                # Clean the extracted text
                full_text = clean_extracted_text(full_text)

                if not full_text.strip() or len(full_text.strip()) < 200:
                    print(f"  ❌ SKIPPED (scanned or empty PDF): {rel_path}")
                    continue

                doc = Document(
                    page_content=full_text,
                    metadata={
                        "source": rel_path,
                        "total_pages": total_pages,
                        "doc_type": doc_type,
                    }
                )
                regular_documents.append(doc)
                print(f"  [OK] Successfully processed {rel_path} ({total_pages} pages, {len(full_text)} chars)")

        except Exception as e:
            print(f"  [ERROR] Error processing {rel_path}: {str(e)}")
            continue

    return regular_documents, timetable_chunks


def load_txt_documents(directory):
    """Load and read text from .txt files, recursively scanning all subfolders.

    Each TXT file becomes a Document with metadata tracking the
    source path and document type.
    """
    documents = []

    # Recursively discover all TXT files
    txt_files = []  # list of (full_path, relative_path)
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith('.txt'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                txt_files.append((full_path, rel_path))

    if not txt_files:
        print("  No TXT files found in the data directory (including subfolders).")
        return documents

    for full_path, rel_path in txt_files:
        try:
            print(f"Processing {rel_path}...")

            # Skip files that are actually binary (e.g., PDFs renamed to .txt)
            if is_binary_file(full_path):
                print(f"  WARNING: {rel_path} appears to be a binary file (PDF?), skipping. Rename to .pdf instead.")
                continue

            # Read with UTF-8 encoding (fallback to latin-1 if needed)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            except UnicodeDecodeError:
                with open(full_path, 'r', encoding='latin-1') as f:
                    full_text = f.read()

            # Clean the extracted text
            full_text = clean_extracted_text(full_text)

            if not full_text.strip():
                print(f"  WARNING: {rel_path} is empty or contains no readable text.")
                continue

            # Create a Document object with metadata
            doc_type = classify_document(rel_path)
            if doc_type is None:
                print(f"  [SKIP] Skipping unsupported TXT file: {rel_path}")
                continue

            doc = Document(
                page_content=full_text,
                metadata={
                    "source": rel_path,
                    "total_pages": 1,
                    "doc_type": doc_type,
                }
            )
            documents.append(doc)
            print(f"  [OK] Successfully processed {rel_path} ({len(full_text)} chars)")

        except Exception as e:
            print(f"  [ERROR] Error processing {rel_path}: {str(e)}")
            continue

    return documents


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def main():
    """Main vectorization pipeline: load PDFs & TXTs → chunk → embed → store in ChromaDB."""

    # Ensure directories exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your institutional PDF/TXT files here.")
        return

    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")
        print("Created 'vector_db_dir' directory for storing vectorized documents.")

    try:
        # Load embedding model
        print("=" * 60)
        print("SMART CAMPUS ASSISTANT -- Document Vectorization")
        print("=" * 60)
        print("\n[1/5] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load and process documents (PDFs + TXTs) recursively from data/
        print("\n[2/5] Loading and processing documents from ./data/ (recursive)...")

        print("\n  --- PDF files ---")
        regular_pdf_docs, timetable_day_chunks = load_pdf_documents("data")

        print("\n  --- TXT files ---")
        txt_documents = load_txt_documents("data")

        # Documents that need generic text splitting
        documents_to_split = regular_pdf_docs + txt_documents

        total_loaded = len(documents_to_split) + len(timetable_day_chunks)
        if total_loaded == 0:
            print("\nNo documents were successfully processed.")
            print("Please add PDF or TXT files to the ./data/ directory and try again.")
            return

        print(f"\n  Regular documents: {len(documents_to_split)} ({len(regular_pdf_docs)} PDF, {len(txt_documents)} TXT)")
        print(f"  Timetable chunks (pre-built): {len(timetable_day_chunks)}")

        # Split regular documents into chunks
        print("\n[3/5] Splitting regular documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=[
                "\n\n--- Page",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]
        )

        if documents_to_split:
            split_chunks = text_splitter.split_documents(documents_to_split)
            total_before = len(split_chunks)

            # Filter out noisy/binary chunks
            split_chunks = [c for c in split_chunks if not is_noisy_chunk(c.page_content)]
            filtered_count = total_before - len(split_chunks)

            print(f"  Split into {total_before} chunks")
            if filtered_count > 0:
                print(f"  Filtered out {filtered_count} noisy/binary chunks")
            print(f"  Clean chunks retained: {len(split_chunks)}")
        else:
            split_chunks = []
            print("  No regular documents to split.")

        # Merge: generic split chunks + pre-built timetable chunks
        print("\n[4/5] Merging all chunks...")
        all_chunks = split_chunks + timetable_day_chunks
        print(f"  Generic split chunks: {len(split_chunks)}")
        print(f"  Timetable chunks: {len(timetable_day_chunks)}")
        print(f"  Total chunks for vector DB: {len(all_chunks)}")

        # Display chunk distribution per source and type
        source_counts = {}
        type_counts = {}
        for chunk in all_chunks:
            src = chunk.metadata.get("source", "unknown")
            day = chunk.metadata.get("day", "")
            doc_type = chunk.metadata.get("doc_type", "unknown")

            label = f"{src} [{day}]" if day else f"{src} [{doc_type}]"
            source_counts[label] = source_counts.get(label, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        print("\n  Chunks per source / type:")
        for src, count in sorted(source_counts.items()):
            print(f"    - {src}: {count} chunk(s)")

        print("\n  Chunks by doc_type:")
        for dtype, count in sorted(type_counts.items()):
            print(f"    - {dtype}: {count}")
            
        # Sanitize metadata before storing in Chroma
        print("\n  Sanitizing metadata for Chroma compatibility...")

        for chunk in all_chunks:
            safe_metadata = {}

            for key, value in chunk.metadata.items():
                key = str(key)

                if key.startswith("_"):
                    continue

                if isinstance(value, (list, dict, tuple, set)):
                    value = str(value)

                safe_metadata[key] = value

            chunk.metadata = safe_metadata

        print("  Metadata sanitization complete.")

        # Create and persist vector database
        print("\n[5/5] Creating vector database in ./vector_db_dir/ ...")

        texts = [doc.page_content for doc in all_chunks]
        metadatas = [doc.metadata for doc in all_chunks]

        import chromadb
        from chromadb.config import Settings

        db_path = "vector_db_dir"

        def _create_and_populate(path):
            """Instantiate ChromaDB, create collection, and add all chunks."""
            client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
            print(f"[DEBUG] Opening fresh Chroma collection v3")
            print(f"[DEBUG] Persist directory path: {path}")

            # Drop old collection if it exists
            try:
                client.delete_collection("campus_documents_v3")
            except Exception:
                pass

            collection = client.get_or_create_collection("campus_documents_v3")
            print(f"[DEBUG] Collection created successfully")

            embeddings_list = embeddings.embed_documents(texts)
            ids = [f"doc_{i}" for i in range(len(texts))]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings_list,
            )
            return collection

        try:
            _create_and_populate(db_path)
        except (KeyError, Exception) as first_err:
            if "_type" in str(first_err) or "configuration" in str(first_err).lower():
                print(f"\n  [!] Incompatible vector DB format detected -- wiping {db_path}/ and retrying...")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path, ignore_errors=True)
                os.makedirs(db_path, exist_ok=True)
                _create_and_populate(db_path)
            else:
                print("TRACEBACK:")
                print(traceback.format_exc())
                raise

        # Compute category-specific counts for the final report
        report_cats = ["syllabus", "regulations", "timetable", "notices", "placements"]
        final_counts = {cat: 0 for cat in report_cats}
        for chunk in all_chunks:
            dtype = str(chunk.metadata.get("doc_type", "unknown")).lower()
            if "timetable" in dtype:
                final_counts["timetable"] += 1
            elif dtype in final_counts:
                final_counts[dtype] += 1

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Vectorized {len(all_chunks)} chunks into vector DB.")
        print("\nChunks by category:")
        print(f"  - Syllabus: {final_counts['syllabus']}")
        print(f"  - Regulations: {final_counts['regulations']}")
        print(f"  - Timetable: {final_counts['timetable']}")
        print(f"  - Notices: {final_counts['notices']}")
        print(f"  - Placements: {final_counts['placements']}")
        print(f"\nVector database stored in ./vector_db_dir/")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print(traceback.format_exc())
        print("\nPlease ensure all required dependencies are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()

