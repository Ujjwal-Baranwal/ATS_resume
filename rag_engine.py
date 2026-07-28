import os
import tempfile

from streamlit import context

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize local models
embeddings = OllamaEmbeddings(model="embeddinggemma")


def build_vector_store(uploaded_files):
    """
    Parses multiple PDF files, attaches source metadata, 
    and indexes all chunks into ChromaDB.
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # 1. Loop through each uploaded file
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # # Add human-readable filename metadata to each document
            for doc in documents:
                doc.metadata["source_filename"] = uploaded_file.name

            # Chunk the documents
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 2. Build Chroma vector store containing chunks from ALL uploaded PDFs
    # Clear old vectors before writing new ones
    vector_db = Chroma(
        collection_name="multi_resume_gemma_chunks",
        embedding_function=embeddings,
    )
    vector_db.delete_collection() # Clears existing duplicates

    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="multi_resume_gemma_chunks"
    )
    # Check total unique chunks in the database
    collection = vector_db._collection
    print(f"Total stored vector items in Chroma: {collection.count()}")
    return vector_db


# def ask_candidate_pool(vector_db, job_description, user_query, top_k=3):
#     """
#     Queries across all candidates in the vector store and ranks them 
#     based on the user's specific question.
#     """
#     # Increase k so we fetch relevant chunks across multiple candidate files
#     retriever = vector_db.as_retriever(search_type='mmr' ,search_kwargs={"k": top_k})

#     system_prompt = (
#         "You are an expert Executive Technical Recruiter comparing candidate resumes against a Target Job Description.\n\n"
#         "--- TARGET JOB DESCRIPTION ---\n"
#         "{job_description}\n\n"
#         "--- RETRIEVED CANDIDATE RESUME CONTEXTS ---\n"
#         "{context}\n\n"
#         "INSTRUCTIONS:\n"
#         "1. Identify all unique candidates by their source filename (e.g., candidate_a.pdf, candidate_b.pdf).\n"
#         "2. Evaluate each candidate's strengths and missing qualifications against the target Job Description.\n"
#         "3. Explicitly declare the **#1 BEST CANDIDATE** for this job description and explain why.\n"
#         "4. Provide a ranked list (1st, 2nd, 3rd place) with estimated fit percentages (e.g., 85% match) and brief justification with direct resume evidence."
#     )


    

#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}")
#     ])

#     partial_prompt = prompt_template.partial(job_description=job_description)

#     combine_docs_chain = create_stuff_documents_chain(llm, partial_prompt)
#     rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

#     user_message = user_query if user_query else "Which candidate is the best fit for this job description and why?"

#     response = rag_chain.invoke({"input": user_message})
#     retrieved_docs = response["context"]

#     # 2. Loop through and inspect the metadata
#     print(f"Total chunks retrieved: {len(retrieved_docs)}\n")

#     for i, doc in enumerate(retrieved_docs):
#         filename = doc.metadata.get("source_filename", "Unknown")
#         page_num = doc.metadata.get("page", "N/A")
        
#         print(f"--- Chunk {i+1} ---")
#         print(f"Candidate File: {filename}")
#         print(f"PDF Page: {page_num}")
#         print(f"Content Sample: {doc.page_content[:80]}...\n")
    
#     return response["answer"], response["context"]

def run_ats_rag(vector_db, job_description, prompt_mode, temperature, custom_system_prompt=None):
    """Retrieves relevant resume context and executes the LLM chain."""
    # Retrieve top 5 most relevant resume chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # Define system instructions based on task
    system_prompts = {
        "coverletter": "You are a Professional Cover letter writer expert for data science roles. Generate a concise cover letter for the candidate based on the Job Description and their resume context.",
        "analysis": "You are a Senior Technical Recruiter. Analyze the candidate resume context against the Job Description. List Strengths, Weaknesses, and Matching Experience.",
        "skills": "You are a Technical Lead. Identify missing hard/soft skills, missing keywords, and suggest specific projects or certifications to close the gap.",
        "match": "You are an ATS Scoring Engine. Evaluate the candidate against the Job Description. Output an Estimated Match Percentage (e.g., 75%), followed by Key Missing Qualifications.",
        # --- NEW PROMPTS ADDED BELOW ---
        "interview_questions": "You are a Hiring Manager. Based on the candidate's gaps relative to the Job Description, generate 5 targeted interview questions with expected answer criteria.",
        "executive_summary": "You are a VP of Engineering. Provide a concise 3-bullet executive summary on whether to pass or fail this candidate for an interview.",
        "salary_alignment": "You are a Compensation Analyst. Compare the candidate's years of experience and level against the job requirements to estimate seniority fit."
    }

    if prompt_mode == "custom" and custom_system_prompt:
        # Allows user to pass a completely custom text prompt dynamically from UI
        selected_system_instruction = custom_system_prompt
    else:
        # Fetches from dictionary, falls back gracefully to 'analysis' if key doesn't exist
        selected_system_instruction = system_prompts.get(prompt_mode, system_prompts["analysis"])
        print("-----------selected_system_instruction-----------", selected_system_instruction)
    system_message = (
        f"{selected_system_instruction}\n\n"
        "JOB DESCRIPTION:\n{job_description}\n\n"
        "RESUME CONTEXT:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Provide a detailed, structured answer based on the retrieved context.")
    ]).partial(job_description=job_description)
    print("-----------prompt_template-----------", prompt_template)
    print("-----------retriever-----------", retriever)
    llm = ChatOllama(model="gemma3", temperature=temperature)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = rag_chain.invoke({"input": job_description})
    
    # Return both the final answer and the retrieved context chunks (needed for Ragas evaluation)
    return response["answer"], response["context"]





def load_full_resumes(uploaded_files):
    """
    Parses each uploaded PDF completely without chunking 
    and returns a combined context string with clear candidate demarcations.
    """
    full_context = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Combine all pages for a single candidate resume
            candidate_text = "\n".join([doc.page_content for doc in documents])
            
            # Format entry with explicit candidate filename tag
            formatted_entry = f"=== CANDIDATE RESUME: {uploaded_file.name} ===\n{candidate_text}\n"
            full_context.append(formatted_entry)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return "\n\n".join(full_context)


# Dynamic function name & updated dynamic system prompt
def compare_candidates(uploaded_files, job_description, user_query, llm):
    """
    Passes full text of up to 5 resumes directly into the LLM context.
    """
    # 1. Load full resume text
    combined_resumes = load_full_resumes(uploaded_files)

    # 2. Build flexible comparison prompt for dynamic candidate counts
    system_prompt = (
        "You are an expert Executive Technical Recruiter comparing candidate resumes against a Target Job Description.\n"
        "You will be given the FULL text of candidate resumes.\n\n"
        "--- TARGET JOB DESCRIPTION ---\n"
        "{job_description}\n\n"
        "--- ALL CANDIDATE RESUMES ---\n"
        "{context}\n\n"
        "INSTRUCTIONS:\n"
        "1. You MUST evaluate ALL candidates provided individually by their exact filename.\n"
        "2. Break down each candidate's Key Strengths, Missing Qualifications, and Overall Fit Percentage (e.g., 85%).\n"
        "3. Rank all uploaded candidates in order from best fit to lowest fit (e.g., 1st Place, 2nd Place, 3rd Place, etc.).\n"
        "4. Explicitly declare the **#1 BEST CANDIDATE** for this job description and justify why with direct evidence from their resume."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    user_message = user_query if user_query.strip() else "Compare all uploaded candidates against the job description and rank them."


    chain = prompt_template | llm
    
    response = chain.invoke({
        "job_description": job_description,
        "context": combined_resumes,
        "input": user_message
    })

    return response.content