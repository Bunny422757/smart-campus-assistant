# Smart Campus Assistant — Dynamic Router Chain (V2 Semantic)
# Orchestration layer: pure MMR retrieval mapping to post-retrieval strict prompting.

import re
from typing import Optional
from langchain_groq import ChatGroq
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.timetable_prompt import TIMETABLE_RULES
from prompts.syllabus_prompt import SYLLABUS_RULES
from prompts.regulation_prompt import REGULATION_RULES

class DynamicRouterChain:
    def __init__(self, vectorstore, college_name, college_short):
        self.vectorstore = vectorstore
        self.college_name = college_name
        self.college_short = college_short
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        )
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1000
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def _condense_question(self, question, chat_history):
        history_str = ""
        for msg in chat_history:
            role = "User" if msg.type == "human" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
            
        prompt = PromptTemplate(
            template="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{history}\n\nFollow Up Input: {question}\nStandalone question: ",
            input_variables=["history", "question"]
        )
        prompt_val = prompt.format(history=history_str, question=question)
        response = self.llm.invoke([HumanMessage(content=prompt_val)])
        answer = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        return answer

    def _doc_type_vote(self, docs):
        """Post-retrieval Layer: Pure majority vote based strictly on doc_type metadata."""
        if not docs:
            return "regulation_mode", 0.0
            
        counts = {}
        for doc in docs:
            dtype = str(doc.metadata.get("doc_type", "regulations")).lower()
            
            # Map raw metadata string to 5 internal router modes
            if "syllabus" in dtype:
                mode_key = "syllabus_mode"
            elif "timetable" in dtype:
                mode_key = "timetable_mode"
            elif "regulation" in dtype or "conduct" in dtype:
                mode_key = "regulation_mode"
            elif "notice" in dtype:
                mode_key = "notice_mode"
            elif "placement" in dtype:
                mode_key = "placement_mode"
            else:
                mode_key = "regulation_mode"
                
            counts[mode_key] = counts.get(mode_key, 0) + 1

        mode = max(counts, key=counts.get)
        confidence = counts[mode] / len(docs)
        return mode, confidence

    def _get_system_prompt(self, mode):
        base_formatting = f"""You are the **Smart Campus Assistant** for **{self.college_name}**.
You answer student questions **strictly based on the provided context**.

### STRICT OUTPUT FORMATTING & STYLE (VERY IMPORTANT)
- **NO REASONING TEXT**: NEVER display `<think>`, reasoning steps, internal analysis, or chain-of-thought. Return ONLY the clean final answer.
- **CONCISE & DIRECT**: Answer ONLY the specific question the student asked. Do not add extra explanations, commentary, or follow-up Q&A pairs unless explicitly requested. Stop immediately after answering.
- **BULLET POINTS**: Use bullet points only when listing items.
- **NOT FOUND**: If the answer is not found in the provided context, respond exactly with: "This information is not available in the campus documents currently uploaded. Please contact the {self.college_short} administration for further details."

### SUBJECT-AWARE ANSWERING (CRITICAL)
Academic PDFs may contain multiple subjects.
1. Always identify the subject mentioned in the question.
2. Use ONLY the context belonging to that subject.
3. Ignore all other subjects in the same PDF. Never mix content across subjects.
"""

        if mode == "timetable_mode":
            return base_formatting + "\n" + TIMETABLE_RULES
        elif mode == "syllabus_mode":
            return base_formatting + "\n" + SYLLABUS_RULES
        elif mode == "regulation_mode":
            return base_formatting + "\n" + REGULATION_RULES
        else:
            return base_formatting + "\n### SEARCH INSTRUCTIONS\nExtract relevant facts to answer directly from the provided context."

    def _extract_subject_from_query(self, question: str) -> tuple[Optional[str], float]:
        """Find subject by word overlap or acronym, now returns (subject, confidence)."""
        if not hasattr(self, '_subject_list'):
            try:
                all_meta = self.vectorstore._collection.get(include=["metadatas"])
                subjects = set()
                for meta in all_meta.get("metadatas", []):
                    subj = meta.get("subject")
                    if subj and subj.lower() != "unknown subject":
                        # Fix common typos found in PDF extraction
                        subj = re.sub(r'\bMA\s+NAGEMENT\b', 'MANAGEMENT', subj, flags=re.IGNORECASE)
                        subj = re.sub(r'\bCOMP\s+ILER\b', 'COMPILER', subj, flags=re.IGNORECASE)
                        subjects.add(subj.strip())
                self._subject_list = list(subjects)
                print(f"[ROUTER] Loaded {len(self._subject_list)} subjects")
            except Exception as e:
                print(f"[ROUTER] Error loading subjects: {e}")
                return None, 0.0

        q_lower = question.lower()
        q_words = set(re.findall(r'[a-z0-9]+', q_lower))
        best_subj = None
        best_score = 0

        for subj in self._subject_list:
            s_lower = subj.lower()
            s_words = set(re.findall(r'[a-z0-9]+', s_lower))
            # 20 points per matched keyword
            score = len(q_words.intersection(s_words)) * 20
            
            # Acronym bonus (40 points)
            acronym = ''.join(w[0] for w in s_words if w).lower()
            if len(acronym) >= 2 and re.search(r'\b' + re.escape(acronym) + r'\b', q_lower):
                score += 40
            
            if score > best_score:
                best_score = score
                best_subj = subj

        # Compute confidence
        q_words_for_conf = set(re.findall(r'[a-z0-9]+', question.lower()))
        max_possible_score = (len(q_words_for_conf) * 20) + 40

        if max_possible_score == 0:
            confidence = 0.0
        else:
            confidence = best_score / max_possible_score

        print(f"[ROUTER] Subject: {best_subj}, confidence: {confidence:.2f}")
        return best_subj, confidence

    def _detect_syllabus_query_type(self, question: str) -> str:
        """
        Classify syllabus question into one of:
        'unit_query', 'book_query', 'topics_query', 'full_syllabus'
        """
        q_lower = question.lower()
        
        # 1. unit_query: contains "unit" followed by a digit or roman numeral i-v
        if re.search(r'\bunit\s*[1-5]\b', q_lower) or re.search(r'\bunit\s*(i|ii|iii|iv|v)\b', q_lower):
            return 'unit_query'
        
        # 2. book_query: contains textbook/reference keywords
        if re.search(r'\b(textbooks?|references?|books?|author|authors)\b', q_lower):
            return 'book_query'
        
        # 3. topics_query: contains topics-related phrases
        if re.search(r'\b(topics?|main topics?|what is taught|what is covered|covered in)\b', q_lower):
            return 'topics_query'
        
        # 4. full_syllabus: default
        return 'full_syllabus'

    def invoke(self, inputs):
        question = inputs["question"]
        chat_vars = self.memory.load_memory_variables({})
        chat_history = chat_vars.get("chat_history", [])
        category = inputs.get("category", "Timetable")
        section = inputs.get("section", "All")
        semester = inputs.get("semester", "All")

        if chat_history and len(question.split()) < 6:
            standalone_question = self._condense_question(question, chat_history)
        else:
            standalone_question = question

        filter_dict = None
        if category == "Timetable":
            # REVISION: Broaden doc_type coverage for Timetable while keeping specific filters
            target_doc_types = [
                "timetable_day",
                "timetable_faculty",
                "timetable_metadata",
                "timetable_almanac"
            ]
                
            filter_conditions = [
                {"doc_type": {"$in": target_doc_types}}
            ]
            
            if section != "All":
                filter_conditions.append({"section": section})
                
            if semester != "All":
                filter_conditions.append({"semester": semester})
                
            day_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', standalone_question, re.IGNORECASE)
            if day_match:
                filter_conditions.append({"day": day_match.group(1).upper()})
                
            if len(filter_conditions) > 1:
                filter_dict = {"$and": filter_conditions}
            else:
                filter_dict = filter_conditions[0]
                
        elif category == "Syllabus":
            detected_subject, confidence = self._extract_subject_from_query(standalone_question)
            if confidence < 0.75:
                print(f"[ROUTER] Subject confidence too low ({confidence:.2f}), falling back to broad syllabus search.")
                detected_subject = None
            
            query_type = self._detect_syllabus_query_type(standalone_question)
            print(f"[ROUTER] Syllabus query type: {query_type}")

            if detected_subject:
                filter_dict = {"$and": [{"doc_type": "syllabus"}, {"subject": detected_subject}]}
            else:
                filter_dict = {"doc_type": "syllabus"}
            print(f"[ROUTER] Syllabus filter: {filter_dict}")
        elif category == "Regulations":
            filter_dict = {"doc_type": "regulations"}
        elif category == "Notices":
            filter_dict = {"doc_type": "notices"}
        elif category == "Placements":
            filter_dict = {"doc_type": "placements"}

        # Pure semantic retrieval
        print(f"[ROUTER V2] Querying Vectorstore for: '{standalone_question}' with Category: '{category}'")
        
        if filter_dict:
            
            custom_k = 8
            docs = self.vectorstore.max_marginal_relevance_search(
                standalone_question,
                k=custom_k,
                fetch_k=30,
                filter=filter_dict
            )
        else:
            # REVISION: No global fallback - The system must strictly follow UI category
            print(f"[ROUTER V2] No valid filter for Category: '{category}'. Returning no documents.")
            docs = []
        
        if not docs:
            answer = f"This information is not available in the campus documents currently uploaded. Please contact the {self.college_short} administration for further details."
            return {"answer": answer, "source_documents": []}

        # Remove noisy chunks and limit results (as per valid cleanup rules)
        docs = [d for d in docs if len(d.page_content.strip()) > 100][:6]


        # Post-retrieval doc verification for Prompt Formatting Strategy
        mode, confidence = self._doc_type_vote(docs)
        print(f"[ROUTER V2] Extracted Style Prompt: {mode} ({confidence:.2f})")

        # LLM Pipeline Strategy
        system_prompt = self._get_system_prompt(mode)
        
        context = "\n\n".join([d.page_content for d in docs])
        history_str = ""
        for msg in chat_history:
            role = "User" if msg.type == "human" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
            
        final_prompt = f"""Context:
{context}

---
Chat History:
{history_str}

---
Student Question: {standalone_question}

Instructions: Answer ONLY the above question using the context provided. Be concise. Stop after answering.
Answer:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt)
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        self.memory.save_context({"question": question}, {"answer": answer})
        
        return {"answer": answer, "source_documents": docs}
