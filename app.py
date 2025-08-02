import streamlit as st
import fitz  # PyMuPDF
import os
import google.generativeai as genai
import faiss
import numpy as np
import re
from io import BytesIO
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY is not set. Please define it as an environment variable or in a .env file.")
genai.configure(api_key=API_KEY)

# PAGE CONFIG 
st.set_page_config(
    page_title="SmartClaim RAG Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    .main-header {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert > div {
        background-color: #f0f8ff;
        border: 1px solid #4facfe !important;
        border-radius: 10px;
        padding: 1rem;
    }
    .success-box, .warning-box, .info-box {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-box { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .warning-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .info-box { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .result-approved {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .result-rejected {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .result-pending {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        cursor: pointer;
    }
    .bookmark-item {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .history-item {
        background: #fff;
        border: 1px solid #dee2e6;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        cursor: pointer;
    }
    .history-item:hover {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION 
# Basic RAG states
for key in ['processed_chunks', 'embeddings', 'valid_chunks', 'faiss_index', 'document_processed']:
    if key not in st.session_state:
        st.session_state[key] = None

# Query and answer states
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'current_chunks' not in st.session_state:
    st.session_state.current_chunks = []

# Search History: Store all previous queries and answers
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Bookmarks: Store important policy sections user wants to save
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []

#HEADER 
st.markdown("""
<div class="main-header">
    <h1>🔍SmartClaim RAG Engine</h1>
    <p>Upload an insurance policy, ask coverage questions, and get grounded AI answers with smart features</p>
</div>
""", unsafe_allow_html=True)

# HELPER FUNCTION FOR SIDEBAR SPACING 
def add_vertical_space(num_lines):
    """
    Add vertical spacing in the sidebar for better layout
    This makes the sidebar look cleaner and more organized
    """
    for _ in range(num_lines):
        st.write("")

def print_creator_info():
    """
    Display creator information in a nicely formatted way
    You can customize this with your own name and details
    """
    creator_info = """
    Prerita Saini✨
    """
    title = "**Created By -**\n\n"
    return title + creator_info

#SIDEBAR (collapsed)
with st.sidebar:
    st.header("💫About")
    st.write("""
Welcome to the **Smart Insurance Policy RAG System**!

This RAG System, helps you understand your insurance policy better. Get instant answers about coverage, exclusions, and benefits using cutting-edge AI technology.
""")

    add_vertical_space(2)

    # System Behavior Settings
    st.markdown("### 🔧 System Behavior")
    st.info("System automatically manages chunking and retrieval settings. You can customize below if needed.")
    advanced = st.checkbox("Show advanced retrieval settings", value=False)
    if advanced:
        st.markdown("#### Retrieval Presets")
        retrieval_preset = st.selectbox("Preset", ["Balanced", "Conservative (fewer chunks)", "Deep Dive (more context)"], index=0)
        if retrieval_preset == "Balanced":
            DEFAULT_TOP_K = 3
            DEFAULT_CHUNK_SIZE = 600
        elif retrieval_preset == "Conservative (fewer chunks)":
            DEFAULT_TOP_K = 1
            DEFAULT_CHUNK_SIZE = 800
        else:
            DEFAULT_TOP_K = 5
            DEFAULT_CHUNK_SIZE = 500
    else:
        DEFAULT_TOP_K = 3
        DEFAULT_CHUNK_SIZE = 600

    add_vertical_space(1)

    # Key Features
    st.markdown("### Key Features")
    st.markdown("""
- **PDF Document Processing**: Upload any insurance policy in PDF format
- **Smart Question Suggestions**: Pre-built common insurance questions
- **Search History**: Keep track of all your previous queries  
- **Bookmark System**: Save important policy sections for quick reference
- **Semantic Search**: AI-powered search through policy documents
- **PDF Report Generation**: Download professional analysis reports
- **Quick Actions**: One-click access to popular insurance scenarios
""")

    add_vertical_space(1)

    # Search History
    st.markdown("### 📚 Search History")
    if st.session_state.search_history:
        st.write(f"Total searches: {len(st.session_state.search_history)}")
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.write("No search history yet.")

    add_vertical_space(1)

    # Bookmarks
    st.markdown("### 🔖 Bookmarks")
    if st.session_state.bookmarks:
        st.write(f"Saved sections: {len(st.session_state.bookmarks)}")
        if st.button("🗑️ Clear Bookmarks"):
            st.session_state.bookmarks = []
            st.rerun()
    else:
        st.write("No bookmarks yet.")

    add_vertical_space(3)

    # Creator Info
    st.sidebar.success(print_creator_info())

#DOCUMENT PROCESSING HELPERS
@st.cache_data
def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from PDF file
    Simple function that opens PDF and gets text from each page
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, max_chars=600):
    """
    Break large text into smaller chunks for better processing
    This helps the AI understand specific parts of the document
    """
    paragraphs = text.split('\n\n')
    if len(paragraphs) < 5:
        paragraphs = text.split('\n')
    if len(paragraphs) < 10:
        paragraphs = re.split(r'[.!?]+\s+', text)

    chunks = []
    current = ""
    for para in paragraphs:
        if not para.strip():
            continue
        if len(current) + len(para) + 1 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para.strip() + " "
        else:
            current += para.strip() + " "
    if current.strip():
        chunks.append(current.strip())

    refined = []
    for chunk in chunks:
        if 50 <= len(chunk) <= max_chars:
            refined.append(chunk)
        elif len(chunk) > max_chars:
            words = chunk.split()
            temp = ""
            for w in words:
                if len(temp) + len(w) + 1 <= max_chars:
                    temp += w + " "
                else:
                    if temp.strip():
                        refined.append(temp.strip())
                    temp = w + " "
            if temp.strip():
                refined.append(temp.strip())
    return refined

def get_embeddings(texts):
    """
    Convert text chunks into numerical vectors (embeddings)
    This allows us to find similar content using math
    """
    embeddings = []
    valid_chunks = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(texts)
    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        if len(text.encode('utf-8')) > 35000:
            continue
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text.strip(),
                task_type="retrieval_document",
                title="Policy Clause"
            )
            embeddings.append(response["embedding"])
            valid_chunks.append(text)
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Embedding chunk {i+1}/{total}")
        except Exception as e:
            st.warning(f"Embedding error on chunk {i}: {e}")
            continue

    progress_bar.empty()
    status_text.empty()

    if not embeddings:
        raise ValueError("No embeddings were generated.")
    return np.array(embeddings), valid_chunks

def build_faiss_index(embeddings):
    """
    Create a search index for fast similarity search
    FAISS is a library that makes searching through embeddings very fast
    """
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def semantic_search(query, valid_chunks, index, k=3):
    """
    Find the most relevant chunks for a user's question
    Uses mathematical similarity to find matching content
    """
    query_emb = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )["embedding"]
    D, I = index.search(np.array([query_emb]).astype('float32'), k=min(k, len(valid_chunks)))
    results = [valid_chunks[i] for i in I[0]]
    return results

def answer_query(query, top_chunks):
    """
    Generate an answer using AI based on relevant policy chunks
    Takes the user's question and relevant policy text to create a helpful answer
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    chunk_text_combined = ""
    for i, chunk in enumerate(top_chunks[:5]):
        snippet = chunk if len(chunk) <= 800 else chunk[:800] + "..."
        chunk_text_combined += f"Chunk {i+1}:\n{snippet}\n\n"

    prompt = f"""You are an assistant helping interpret insurance policy language. Use the following policy text to answer the query.

Policy Text:
{chunk_text_combined}

Query: {query}

Provide:
1. Decision: Approved / Rejected / Need More Info
2. Reasoning: Short explanation citing the relevant policy language
3. Payout amount: If determinable, else say 'Not enough information'
4. Relevant clause: Identify which part of the provided chunks supports your answer.

If the supplied chunks are insufficient, explicitly request further context."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"

# NEW FEATURE FUNCTIONS

def get_smart_suggestions():
    """
    Return a list of common insurance questions that users often ask
    This helps users get started without thinking of questions themselves
    """
    return [
        "What is covered for dental treatment?",
        "Does my policy cover knee surgery for a 46-year-old male?",
        "Are pre-existing conditions included?",
        "Emergency room visit coverage details",
        "Maternity benefits eligibility",
        "What is my deductible amount?",
        "Coverage for prescription medications",
        "Mental health treatment coverage",
        "Physical therapy coverage limits",
        "Coverage for medical equipment like wheelchairs"
    ]

def add_to_search_history(query, answer):
    """
    Save the user's question and answer to history
    This lets users revisit previous searches easily
    """
    history_item = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'answer': answer[:200] + "..." if len(answer) > 200 else answer  # Store short version
    }
    # Add to beginning of list (most recent first)
    st.session_state.search_history.insert(0, history_item)
    # Keep only last 20 searches to avoid memory issues
    if len(st.session_state.search_history) > 20:
        st.session_state.search_history = st.session_state.search_history[:20]

def add_bookmark(chunk_text, query):
    """
    Save important policy sections that user wants to remember
    Like bookmarking a webpage, but for policy sections
    """
    bookmark = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'text': chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
        'full_text': chunk_text
    }
    st.session_state.bookmarks.insert(0, bookmark)
    # It keeps only last 10 bookmarks
    if len(st.session_state.bookmarks) > 10:
        st.session_state.bookmarks = st.session_state.bookmarks[:10]

def generate_pdf_report(query, answer, chunks):
    """
    Create a PDF report with the analysis results
    Users can download and save their insurance analysis
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Insurance Policy Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Generated date
    date = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    story.append(date)
    story.append(Spacer(1, 12))
    
    # Query
    query_title = Paragraph("Your Question:", styles['Heading2'])
    story.append(query_title)
    query_text = Paragraph(query, styles['Normal'])
    story.append(query_text)
    story.append(Spacer(1, 12))
    
    # Answer
    answer_title = Paragraph("Analysis Result:", styles['Heading2'])
    story.append(answer_title)
    answer_text = Paragraph(answer.replace('\n', '<br/>'), styles['Normal'])
    story.append(answer_text)
    story.append(Spacer(1, 12))
    
    # Relevant sections
    sections_title = Paragraph("Relevant Policy Sections:", styles['Heading2'])
    story.append(sections_title)
    
    for i, chunk in enumerate(chunks):
        section_text = Paragraph(f"Section {i+1}: {chunk[:500]}{'...' if len(chunk) > 500 else ''}", styles['Normal'])
        story.append(section_text)
        story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

#HELP & ABOUT
with st.expander("ℹ️ How to Use This Enhanced System"):
    st.markdown("""
    **1. Upload PDF:** Upload your insurance policy.  
    **2. Process:** Click to process—it auto-chunks and creates embeddings.  
    **3. Use Quick Actions:** Click suggested questions for instant answers.  
    **4. Ask Custom Questions:** Type your own questions for specific coverage details.  
    **5. Review History:** Check your previous searches anytime.  
    **6. Bookmark Important Sections:** Save important policy parts for later.  
    **7. Generate Reports:** Download PDF reports of your analysis.

    ### Tips for Best Results:
    - Be specific (include age, condition, policy duration)
    - Use Quick Actions to explore common scenarios
    - Bookmark sections you reference frequently
    - Generate PDF reports for important decisions
    """)

#UPLOAD / PROCESS SECTION
st.markdown("### 📄 Upload Policy Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload your insurance policy document in PDF format")

if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.info(f"📦 Size: {uploaded_file.size / (1024*1024):.2f} MB")

    if st.button("🔄 Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                pdf_bytes = uploaded_file.read()
                text = extract_text_from_pdf(pdf_bytes)
                st.success(f"📝 Extracted {len(text):,} characters from the PDF")

                chunks = chunk_text(text, max_chars=DEFAULT_CHUNK_SIZE)
                st.success(f"📚 Created {len(chunks)} text chunks")

                embeddings, valid_chunks = get_embeddings(chunks)
                index = build_faiss_index(embeddings)

                st.session_state.embeddings = embeddings
                st.session_state.valid_chunks = valid_chunks
                st.session_state.faiss_index = index
                st.session_state.document_processed = True

                st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Document Processed</h3>
                        <p>Generated {len(embeddings)} embeddings and ready to answer queries.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Document processing failed: {e}")
else:
    st.warning("⚠️ Please upload a PDF policy document to begin.")

st.markdown("---")

#Quick Action Buttons
st.markdown("### ⚡ Quick Actions - Common Questions")
st.write("Click any button below to quickly ask common insurance questions:")

#Create columns for better layout of buttons
col1, col2, col3 = st.columns(3)
suggestions = get_smart_suggestions()

#Initialize a flag to track if we just processed a quick action
if 'just_processed_quick_action' not in st.session_state:
    st.session_state.just_processed_quick_action = False

#Display quick action buttons in columns
for i, suggestion in enumerate(suggestions[:9]):  # Show first 9 suggestions
    col = [col1, col2, col3][i % 3]  # Distribute across 3 columns
    with col:
        if st.button(f"{suggestion[:30]}...", key=f"suggestion_{i}"):
            st.session_state.current_query = suggestion
            # Auto-trigger search if document is processed
            if st.session_state.document_processed:
                with st.spinner("Analyzing query..."):
                    try:
                        top_chunks = semantic_search(
                            suggestion,
                            st.session_state.valid_chunks,
                            st.session_state.faiss_index,
                            k=DEFAULT_TOP_K
                        )
                        answer = answer_query(suggestion, top_chunks)
                        
                        st.session_state.current_answer = answer
                        st.session_state.current_chunks = top_chunks
                        
                        # Add to history only once
                        add_to_search_history(suggestion, answer)
                        
                        # Set flag to prevent double processing
                        st.session_state.just_processed_quick_action = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during QA: {e}")

st.markdown("---")
st.markdown("### ❓ Ask Questions")

#Smart Suggestions with Auto-complete
st.markdown("**Popular questions to get you started:**")
with st.expander("View all suggestion categories"):
    st.write("**Coverage Questions:**")
    st.write("• What treatments are covered?")
    st.write("• Are pre-existing conditions included?")
    st.write("• Emergency room coverage details")
    
    st.write("**Financial Questions:**")
    st.write("• What is my deductible?")
    st.write("• Coverage limits and maximums")
    st.write("• Co-payment requirements")
    
    st.write("**Specific Scenarios:**")
    st.write("• Surgery coverage for specific age/condition")
    st.write("• Maternity and pregnancy benefits")
    st.write("• Mental health treatment coverage")

#Primary question entry with current query pre-filled
query = st.text_area(
    "Enter your insurance coverage question:",
    value=st.session_state.current_query,  # Pre-fill with selected suggestion
    placeholder="Example: 46M, knee surgery, 3-month policy. Is pre-existing knee arthritis covered?",
    height=140
)

#Disclaimer
st.info("❗ This system provides informational summaries. It is not a substitute for official policy review or professional advice.")

#Only show the "Get Answer" button if we haven't just processed a quick action or if the user has modified the query
query_changed = query.strip() != st.session_state.current_query
show_get_answer_button = not st.session_state.just_processed_quick_action or query_changed

if show_get_answer_button:
    if st.button("🔍 Get Answer", type="primary", disabled=not query.strip()):
        if not st.session_state.document_processed:
            st.warning("Upload and process a document first.")
        else:
            # Only process if query is different from current or if no answer exists
            if (query.strip() != st.session_state.current_query or 
                not st.session_state.current_answer):
                with st.spinner("Analyzing query..."):
                    try:
                        top_chunks = semantic_search(
                            query,
                            st.session_state.valid_chunks,
                            st.session_state.faiss_index,
                            k=DEFAULT_TOP_K
                        )
                        answer = answer_query(query, top_chunks)

                        st.session_state.current_query = query
                        st.session_state.current_answer = answer
                        st.session_state.current_chunks = top_chunks
                        
                        # Add to search history only if it's a new search
                        add_to_search_history(query, answer)
                        
                    except Exception as e:
                        st.error(f"❌ Error during QA: {e}")
            
            # Reset the flag after processing
            st.session_state.just_processed_quick_action = False
else:
    # Show a message that the query has been processed
    st.success("✅ Question processed! Results are shown below. Modify the question above to ask something different.")
    # Reset the flag when user sees this message
    if st.button("🔄 Ask New Question"):
        st.session_state.just_processed_quick_action = False
        st.session_state.current_query = ""
        st.rerun()

#DISPLAY RESULTS
st.markdown("---")
st.markdown("### 📋 Analysis Results")

if st.session_state.document_processed and st.session_state.current_answer:
    #Interpret decision
    lines = st.session_state.current_answer.lower().splitlines()
    decision_line = next((l for l in lines if "decision" in l), "")
    if "approved" in decision_line:
        status_class = "result-approved"
        emoji = "✅"
        status_text = "APPROVED"
    elif "rejected" in decision_line:
        status_class = "result-rejected"
        emoji = "❌"
        status_text = "REJECTED"
    else:
        status_class = "result-pending"
        emoji = "🟡"
        status_text = "NEEDS REVIEW"

    st.markdown(f"""
    <div class="{status_class}">
        <h3>{emoji} {status_text}</h3>
        <p><strong>Query:</strong> {st.session_state.current_query}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📝 Detailed Answer")
    st.write(st.session_state.current_answer)
    
    # PDF Report Generation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate PDF Report"):
            pdf_buffer = generate_pdf_report(
                st.session_state.current_query,
                st.session_state.current_answer,
                st.session_state.current_chunks
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name=f"insurance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        if st.button("🔖 Bookmark This Analysis"):
            add_bookmark(st.session_state.current_answer, st.session_state.current_query)
            st.success("✅ Added to bookmarks!")

    with st.expander("🔍 Relevant Policy Sections Used"):
        for i, chunk in enumerate(st.session_state.current_chunks):
            st.markdown(f"**Section {i+1}:**")
            st.write(chunk[:500] + ("..." if len(chunk) > 500 else ""))
            
            # Bookmark individual sections
            if st.button(f"🔖 Bookmark Section {i+1}", key=f"bookmark_section_{i}"):
                add_bookmark(chunk, f"Section from: {st.session_state.current_query}")
                st.success(f"✅ Section {i+1} bookmarked!")
                
            st.divider()
else:
    if st.session_state.document_processed:
        st.info("Ask a question to get an answer based on the uploaded policy.")
    else:
        st.info("Upload and process a document first.")

# Search History Display
if st.session_state.search_history:
    st.markdown("---")
    st.markdown("### 📚 Recent Search History")
    st.write("Click any previous search to view it again:")
    
    for i, item in enumerate(st.session_state.search_history[:5]):  # Show last 5 searches
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"🕒 {item['query'][:60]}...", key=f"history_{i}"):
                    st.session_state.current_query = item['query']
                    st.rerun()
            with col2:
                st.write(f"_{item['timestamp']}_")

# Bookmarks Display
if st.session_state.bookmarks:
    st.markdown("---")
    st.markdown("### 🔖 Your Bookmarks")
    st.write("Your saved policy sections and analyses:")
    
    for i, bookmark in enumerate(st.session_state.bookmarks):
        with st.expander(f"📑 {bookmark['query'][:50]}... ({bookmark['timestamp']})"):
            st.write("**Related Query:**", bookmark['query'])
            st.write("**Saved Content:**")
            st.write(bookmark['text'])
            if st.button(f"🗑️ Remove Bookmark", key=f"remove_bookmark_{i}"):
                st.session_state.bookmarks.pop(i)
                st.rerun()

# METRICS / STATS FOOTER 
if st.session_state.document_processed:
    st.markdown("---")
    st.markdown("### 📊 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Chunks", len(st.session_state.valid_chunks))
    with col2:
        st.metric("🧠 Embeddings", len(st.session_state.embeddings))
    with col3:
        avg_size = np.mean([len(c) for c in st.session_state.valid_chunks]) if st.session_state.valid_chunks else 0
        st.metric("📏 Avg Chunk Size", f"{avg_size:.0f} chars")
    with col4:
        st.metric("🔍 Search Ready", "✅ Yes")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666;">
    <p>Enhanced with smart features for better insurance policy analysis 🚀</p>
</div>

""", unsafe_allow_html=True)
