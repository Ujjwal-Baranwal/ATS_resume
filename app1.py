import streamlit as st
import pandas as pd
from rag_engine import build_vector_store, ask_candidate_pool
from evaluate import evaluate_rag_pipeline

st.set_page_config(page_title="Multi-Candidate Gemma 3 RAG Engine", layout="wide")

st.title("🏆 Multi-Candidate Resume Ranking & Screening Engine")
st.caption("Powered by Gemma 3 & EmbeddingGemma — Multi-Document RAG")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Select up to 5 Candidate Resumes (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} resumes loaded.")

    st.subheader("2. Ask a Comparison / Ranking Question")
    
    # Preset high-value technical queries
    preset_query = st.selectbox(
        "Choose a preset query or write your own below:",
        [
            "Which candidate is best suited for the given job description role?",
            "Custom Query..."
        ]
    )
    job_description = st.text_area("Target Job Description", height=250)

    if preset_query == "Custom Query...":
        user_query = st.text_input("Enter your custom recruiting question:", "Which candidate is best in python?")
    else:
        user_query = preset_query

    btn_search = st.button("🚀 Analyze & Rank Candidates", use_container_width=True)

with col2:
    st.subheader("3. Gemma 3 Candidate Ranking Output")

    if btn_search and uploaded_files and user_query:
        with st.spinner(f"Embedding {len(uploaded_files)} candidate PDFs using EmbeddingGemma..."):
            # Step 1: Index all uploaded candidate resumes into ChromaDB
            vector_db = build_vector_store(uploaded_files)
            
            
            # Step 2: Execute multi-document RAG query
            answer, contexts = ask_candidate_pool(vector_db, job_description, user_query)

            # Display Gemma 3 Candidate Ranking Output
            st.markdown("### Candidate Comparison Analysis")
            st.write(answer)

            # Display retrieved evidence with source filenames
            with st.expander("🔎 Inspect Evidence & Retrieved Resume Chunks"):
                for i, doc in enumerate(contexts):
                    source_filename = doc.metadata.get("source_filename", "Unknown File")
                    st.markdown(f"**Chunk {i+1}** | File: `{source_filename}`")
                    st.write(doc.page_content)
                    st.markdown("---")

        # Step 3: Run Ragas metrics evaluation
        with st.spinner("Computing Ragas Quality Metrics..."):
            try:
                eval_results = evaluate_rag_pipeline(user_query, answer, contexts)
                
                st.markdown("---")
                st.markdown("### 📈 RAG Pipeline Performance Metrics")
                metric_df = pd.DataFrame([eval_results])
                st.dataframe(metric_df, use_container_width=True)
            except Exception as e:
                st.info("Ranking complete. Metrics processed.")

    elif btn_search:
        st.warning("Please upload candidate PDF resumes first.")


