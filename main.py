# Smart Campus Assistant — GRIET
# An AI-powered campus assistant using RAG (Retrieval-Augmented Generation)
# that provides accurate, document-backed answers from institutional PDFs.
#
# This is a FastAPI backend combined with a Streamlit frontend using LangChain and Groq LLM.
# config.json must contain your GROQ_API_KEY.
# To run:  streamlit run main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fpdf import FPDF
from datetime import datetime

from router import DynamicRouterChain

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COLLEGE_NAME = "Gokaraju Rangaraju Institute of Engineering and Technology (GRIET)"
COLLEGE_SHORT = "GRIET"

# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def remove_emojis(text):
    """Remove emoji characters from text for clean PDF export."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ──────────────────────────────────────────────
# FastAPI app (REST endpoint for programmatic access)
# ──────────────────────────────────────────────

app = FastAPI(
    title="Smart Campus Assistant API",
    description=f"REST API for {COLLEGE_SHORT} campus chatbot"
)

class MessageRequest(BaseModel):
    message: str

@app.post("/chat")
async def chatbot(request: MessageRequest):
    """Handle chat requests via the REST API."""
    message = request.message

    # Setup vectorstore and conversational chain
    vectorstore = setup_vectorstore()
    conversational_chain = chat_chain(vectorstore)

    # Check for off-topic queries
    if contains_off_topic(message):
        response = (
            f"I'm sorry, but I can only answer questions related to "
            f"{COLLEGE_SHORT} campus information. Please ask about regulations, "
            f"timetables, syllabus, notices, or placements."
        )
    else:
        try:
            response_data = conversational_chain.invoke({
                "question": message,
                "chat_history": []
            })
            response = response_data["answer"]
        except Exception as e:
            if "RateLimit" in type(e).__name__ or "rate limit" in str(e).lower() or "429" in str(e):
                response = "Rate limit reached. Please try again after some time."
            else:
                response = f"An error occurred: {str(e)}"

    return {"response": response}

# ──────────────────────────────────────────────
# Off-topic detection
# ──────────────────────────────────────────────

def contains_off_topic(question):
    """Detect questions clearly outside campus scope."""
    off_topic_keywords = [
        "stock market", "cryptocurrency", "bitcoin",
        "movie review", "recipe", "cook",
        "political party", "election results",
        "dating advice", "relationship",
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in off_topic_keywords)

# ──────────────────────────────────────────────
# Vector store setup
# ──────────────────────────────────────────────

def setup_vectorstore():
    """Initialize ChromaDB vector store with HuggingFace embeddings."""
    persist_directory = f"{working_dir}/vector_db_dir"
    
    print(f"[DEBUG Vectorstore] Binding ChromaDB to persist directory: {persist_directory}")
    
    embeddings = HuggingFaceEmbeddings(
         model_name="sentence-transformers/all-MiniLM-L6-v2"
     )
    
    import chromadb
    from chromadb.config import Settings
    
    # Initialize explicit client to forcefully link to the correct DB without path mangling
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    
    vectorstore = Chroma(
        client=client,
        collection_name="campus_documents_v3",
        embedding_function=embeddings
    )
    return vectorstore

# ──────────────────────────────────────────────
# Conversational RAG chain
# ──────────────────────────────────────────────

def chat_chain(vectorstore):
    """Return the Dynamic Router instance, replacing ConversationalRetrievalChain."""
    return DynamicRouterChain(vectorstore, COLLEGE_NAME, COLLEGE_SHORT)

@st.cache_resource
def load_vectorstore():
    return setup_vectorstore()

@st.cache_resource
def load_chain(_vectorstore):
    return chat_chain(_vectorstore)

# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title=f"Smart Campus Assistant — {COLLEGE_SHORT}",
    page_icon="🎓",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.title("About the Assistant")
    
    # Description
    st.markdown("## Description")
    st.markdown(f"""
        <div class="css-textbarboxtype">
            An AI-powered Smart Campus Assistant for <b>{COLLEGE_SHORT}</b> that provides
            accurate, document-backed answers from institutional PDFs such as
            regulations, timetables, syllabus, notices, and placements.
        </div>
    """, unsafe_allow_html=True)
    
    # Goals
    st.markdown("## Goals")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Student Support<br>
            - Admissions Guidance<br>
            - Regulations<br>
            - Campus Services<br>
            - Timetable Queries<br>
            - Syllabus<br>
        </div>
    """, unsafe_allow_html=True)
    
    # Purpose
    st.markdown("## Purpose")
    st.markdown(f"""
        <div class="css-textbarboxtype">
            Designed as a seamless, user-friendly assistant for {COLLEGE_NAME},
            this chatbot helps students and faculty easily access accurate
            information from official campus documents. Whether you have questions
            about academic regulations, class schedules, examination policies,
            notices, or placements, the assistant retrieves and presents reliable,
            context-aware answers powered by {COLLEGE_SHORT}'s verified knowledge base.
        </div>
    """, unsafe_allow_html=True)
    
    # Values
    st.markdown("## Our Values")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Student-Centered<br>
            - Accessibility<br>
            - Accuracy<br>
            - Transparency<br>
            - Professionalism<br>
            - Inclusivity<br>
            - Excellence<br>
            - Support<br>
            - Integrity<br>
            - Continuous Improvement<br>
        </div>
    """, unsafe_allow_html=True)
    
    # ── Chat History Section ──
    st.markdown("---")
    st.markdown("## Chat History")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history previews
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                st.session_state.selected_chat = idx//2

        # ── Database Controls ──
    st.markdown("---")
    st.markdown("## Database Controls")

    if st.button("🗑️ Clear Database", type="primary"):
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=f"{working_dir}/vector_db_dir",
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            client.delete_collection("campus_documents_v3")
        except Exception:
            pass
        st.cache_resource.clear()
        st.session_state.chat_history = []
        for key in ["conversational_chain", "vectorstore"]:
            st.session_state.pop(key, None)
        st.success("✅ Vector database and chat history cleared.")
        st.rerun()

    if st.button("🔄 Clear and Rebuild Database"):
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=f"{working_dir}/vector_db_dir",
            settings=Settings(anonymized_telemetry=False)
        )
        try:
            client.delete_collection("campus_documents_v3")
        except Exception:
            pass
        st.cache_resource.clear()
        st.session_state.chat_history = []
        for key in ["conversational_chain", "vectorstore"]:
            st.session_state.pop(key, None)
        exit_code = os.system("python vectorize_documents.py")
        if exit_code == 0:
            st.success("✅ Database rebuilt successfully.")
        else:
            st.error(f"❌ Rebuild failed with exit code {exit_code}")
        st.rerun()

    # ── PDF Export ──
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                pdf = FPDF()
                pdf.add_page()
                
                # PDF Header
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, f"{COLLEGE_SHORT} Smart Campus Assistant - Chat History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                # Add conversation
                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    # Role header
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    # Message content
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)
                
                # Save PDF
                filename = f"griet_campus_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)
                
                # Create download button
                with open(filename, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                
                # Clean up the file
                os.remove(filename)
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

# ── Main Chat Interface ──


st.title(f"🎓 {COLLEGE_SHORT} Smart Campus Assistant")
st.caption(f"Ask questions about {COLLEGE_NAME} — regulations, timetables, syllabus, and more.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = load_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Category selection
category_options = {
    "Timetables": "Timetable",
    "Syllabus": "Syllabus",
    "Regulations": "Regulations",
    "Notices": "Notices",
    "Placements": "Placements"
}

selected_label = st.selectbox(
    "Search Category (Optional):",
    list(category_options.keys())
)

selected_category = category_options[selected_label]

selected_section = "All"
selected_semester = "All"
if selected_category == "Timetable":
    sections = set()
    semesters = set()
    try:
        results = st.session_state.vectorstore._collection.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            if meta and "timetable" in str(meta.get("doc_type", "")).lower():
                sec = meta.get("section")
                sem = meta.get("semester")
                if sec: sections.add(str(sec).strip())
                if sem: semesters.add(str(sem).strip())
    except Exception as e:
        pass

    if sections or semesters:
        col1, col2 = st.columns(2)
        
        sorted_sections = sorted(list(sections))
        
        ROMAN_ORDER = {"I":1, "II":2, "III":3, "IV":4, "V":5, "VI":6, "VII":7, "VIII":8}
        def sem_sort_key(sem_str):
            parts = sem_str.split()
            if parts and parts[0].upper() in ROMAN_ORDER:
                return ROMAN_ORDER[parts[0].upper()]
            return 99
            
        sorted_semesters = sorted(list(semesters), key=sem_sort_key)

        with col1:
            if sorted_sections:
                opts = ["All"] + sorted_sections
                def_idx = 1 if len(sorted_sections) == 1 else 0
                selected_section = st.selectbox("Select Section", opts, index=def_idx)
        with col2:
            if sorted_semesters:
                opts = ["All"] + sorted_semesters
                def_idx = 1 if len(sorted_semesters) == 1 else 0
                selected_semester = st.selectbox("Select Semester", opts, index=def_idx)

# Chat input
user_input = st.chat_input(f"Ask a question about {COLLEGE_SHORT}...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.conversational_chain.invoke({
                "question": user_input,
                "chat_history": [],
                "category": selected_category,
                "section": selected_section,
                "semester": selected_semester
            })
        except Exception as e:
            if "RateLimit" in type(e).__name__ or "rate limit" in str(e).lower() or "429" in str(e):
                st.markdown("Rate limit reached. Please try again after some time.")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Rate limit reached. Please try again after some time."
                })
                st.stop()
            else:
                st.markdown(f"An error occurred: {str(e)}")
                st.stop()

        print("\n" + "=" * 80)
        print("DEBUG: USER QUESTION")
        print(user_input)
        print("=" * 80)

        print("\nDEBUG: RETRIEVED SOURCE DOCUMENTS")
        docs = response.get("source_documents", [])

        if not docs:
            print("NO DOCUMENTS RETRIEVED")
        else:
            for i, doc in enumerate(docs, 1):
                print(f"\nDOCUMENT {i}")
                print("METADATA:", doc.metadata)
                print("CONTENT:")
                print(doc.page_content[:1000])
                print("-" * 80)

        assistant_response = response["answer"]

        print("\nDEBUG: FINAL ANSWER")
        print(assistant_response)
        print("=" * 80)

        st.markdown(assistant_response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
