import os
import streamlit as st
import pandas as pd
import numpy as np
from rag_engine import build_vector_store, run_ats_rag
from evaluate import evaluate_rag_pipeline
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "ATS_resume"  # Project name in dashboard

# Sidebar - Model Configuration
st.sidebar.header("Model Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

st.set_page_config(page_title="Multi-Resume RAG ATS Engine", layout="wide")

st.title("📄 Multi-Document Local RAG ATS Engine")
st.caption("Powered by Gemma 3 & EmbeddingGemma")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Data")
    
    # Enable multiple PDF uploads
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes / Documents (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True  # <-- Allows multiple file selection
    )
    job_description = st.text_area("Target Job Description", height=250)

with col2:
    st.subheader("2. Analysis & Evaluation Output")

    # 1. Map user-friendly dropdown options to your backend prompt_mode keys
    PROMPT_OPTIONS = {
        "General Analysis": "analysis",
        "Missing Skills & Gaps": "skills",
        "ATS Match Percentage": "match",
        "Interview Questions": "interview_questions",
        "Executive Summary": "executive_summary",
        "Cover Letter": "coverletter",
        "✍️ Custom System Prompt": "custom"  # Special option for custom input
    }

    st.subheader("Select Evaluation Mode")

    # 2. Selectbox for picking preset modes or custom mode
    selected_label = st.selectbox(
        "Choose evaluation focus:",
        options=list(PROMPT_OPTIONS.keys()),
        index=0
    )

    # Extract internal key (e.g., "analysis", "skills", or "custom")
    prompt_mode = PROMPT_OPTIONS[selected_label]

    # 3. Conditionally display text area ONLY if "Custom System Prompt" is chosen
    custom_system_prompt = None

    if prompt_mode == "custom":
        custom_system_prompt = st.text_area(
            "Enter your custom system instructions:",
            placeholder="Example: You are a Lead Data Scientist. Evaluate the candidate's hands-on experience with PyTorch and distributed training...",
            height=150
        )

    if uploaded_files and job_description:

        with st.spinner("Processing..."):
            # Step 1: Index Vector DB
            vector_db = build_vector_store(uploaded_files)

            

            # 4. Trigger the pipeline on button click
            if st.button("Run ATS Evaluation", type="primary"):
                # Safety validation if custom mode was selected but left empty
                if prompt_mode == "custom" and not custom_system_prompt.strip():
                    st.warning("Please enter custom instructions before running.")
                else:
                    with st.spinner("Analyzing resume..."):
                        answer, contexts = run_ats_rag(
                            vector_db=vector_db,
                            job_description=job_description,
                            prompt_mode=prompt_mode,
                            temperature=temperature,
                            custom_system_prompt=custom_system_prompt
                        )
                        
                        st.success("Evaluation Complete!")
            

                        # Display LLM Output
                        st.markdown("### AI Evaluation")
                        st.write(answer)

                        # Inspect source files alongside context chunks
                        with st.expander("🔎 Inspect Retrieved Context Chunks & Sources"):
                            for i, doc in enumerate(contexts):
                                source_name = doc.metadata.get("source_filename", "Unknown Document")
                                st.markdown(f"**Chunk {i+1}** *(File: `{source_name}`)*:")
                                st.write(doc.page_content)
                                st.markdown("---")

                    # Step 3: Run Ragas Evaluation
                    # with st.spinner("Calculating Ragas Evaluation Metrics..."):
                    #     try:
                    #         eval_results = evaluate_rag_pipeline(job_description, answer, contexts)
                            
                    #         st.markdown("---")
                    #         st.markdown("### 📈 RAG Pipeline Performance Metrics")

                    #         if hasattr(eval_results, "to_pandas"):
                    #             df_results = eval_results.to_pandas()
                    #         else:
                    #             df_results = pd.DataFrame([dict(eval_results)])
                    #         metric_df = df_results.astype(str)
                    #         st.dataframe(metric_df, width='stretch')
                            
                    #     except Exception as e:
                    #         st.info("Evaluation metrics generated. Note: Ground truth is optional for Faithfulness & Context Precision.")

    else:
        st.warning("Please upload a resume PDF and provide a job description.")