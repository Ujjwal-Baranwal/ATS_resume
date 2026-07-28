import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rag_engine import load_full_resumes, compare_candidates
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "ATS_resume"  # Project name in dashboard



# Initialize Streamlit Page Config
st.set_page_config(
    page_title="Resume Ranker & Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Candidate Resume Matcher & Ranker")
st.markdown(
    "Upload **up to 5 candidate resumes (PDFs)** and provide a **Job Description** "
    "to generate a comprehensive side-by-side candidate analysis."
)

# Sidebar - Model Configuration
st.sidebar.header("Model Settings")
model_name = st.sidebar.text_input("Ollama Model", value="gemma3")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Initialize Ollama LLM
llm = ChatOllama(model=model_name, temperature=temperature)

# Streamlit Input UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area(
        "Paste the Job Description here:",
        height=250,
        placeholder="e.g., Seeking a Senior Data Scientist with experience in PyTorch, MLOps, LLM RAG pipelines..."
    )

    st.subheader("2. Custom Prompt / Focus Question (Optional)")
    user_query = st.text_input(
        "Specific evaluation question:",
        value="Which candidate is the best fit for this job description and why?",
        placeholder="e.g., Which candidate has the strongest production deployment experience?"
    )

with col2:
    st.subheader("3. Upload Candidate Resumes")
    # Updated file uploader and validation
    uploaded_files = st.file_uploader(
        "Upload Candidate PDF Resumes (1 to 5 files)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📁 **Uploaded Files ({len(uploaded_files)}):** " + ", ".join([f.name for f in uploaded_files]))
        
        # Enforce maximum limit of 5 files
        if len(uploaded_files) > 5:
            st.error("❌ Maximum limit exceeded. Please upload **no more than 5 resumes** at a time.")

# Action Button
st.markdown("---")
if st.button("🚀 Analyze & Compare Candidates", type="primary", use_container_width=True):
    if not job_description.strip():
        st.error("Please enter a Job Description before running the comparison.")
    elif not uploaded_files:
        st.error("Please upload candidate PDF resumes before running the comparison.")
    elif len(uploaded_files) > 5:
        st.error("Please reduce the number of uploaded resumes to 5 or fewer before proceeding.")
    else:
        with st.spinner(f"Reading {len(uploaded_files)} candidate resume(s) and generating evaluation..."):
            try:
                # Call updated function
                analysis = compare_candidates(uploaded_files, job_description, user_query, llm)
                st.subheader("📊 Candidate Evaluation & Ranking")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")