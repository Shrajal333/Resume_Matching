# Import libraries
import os
import io
import re
import json
import shutil
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from llama_index.core.settings import Settings
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

# Import custom libraries
from dotenv import load_dotenv
from ats.schema import jd_schema
from ats.helper import row_to_text
from tools.model import client_tool
from tools.render import render_candidate
from ats.helper import generate_multiqueries
from tools.file_handler import FileHandlerProcessor
from parsing.resume_processing import process_resumes
from ats.scorer import compute_bm25_filtered_scores, compute_jaccard_filtered_scores, compute_node_scores

# Import env variables and config
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
TOP_N = 15
MIN_RAW_SCORE = 25

client = client_tool()
job_schema = jd_schema()

# Streamlit config
st.set_page_config(page_title="Resume Processing System", page_icon="📄", layout="wide")

# Session state variables
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'last_upload' not in st.session_state:
    st.session_state.last_upload = {}
if 'last_ranking_results' not in st.session_state:
    st.session_state.last_ranking_results = []
if 'last_ranking_jd' not in st.session_state:
    st.session_state.last_ranking_jd = ""
if 'filtered_candidates' not in st.session_state:
    st.session_state.filtered_candidates = None

# Clear Streamlit variables
def clear_results(processor):
    st.session_state.processed_data = None
    st.session_state.processing_complete = False
    st.session_state.processing_time = None
    st.session_state.last_ranking_results = []
    st.session_state.last_ranking_jd = ""
    st.session_state.filtered_candidates = None
    processor.cleanup_temp_files()

# Clear temp directory
def clear_temp_resumes():
    output_dir = Path("temp_resumes")

    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Normalize scores between 1-100
def normalize(arr):
    arr = np.array(arr)
    if np.ptp(arr) < 1e-8:
        # If all values are the same (or only one entry), set to 1
        return np.ones_like(arr)
    return (arr - np.min(arr)) / (np.ptp(arr) + 1e-8)

# Variables
max_workers = 8
processor = FileHandlerProcessor()

# Sidebar Control Panel
with st.sidebar:
    st.title("🛠️ Control Panel")
    st.markdown("**Quickly parse and manage your resume files!**")

    st.divider()
    selected_option = st.radio(
        "Choose Input Method",
        ["📁 File Upload", "🔗 URL/Links", "📦 Zip Upload"],
        index=0,
        help="Select how you'd like to provide resumes for processing.",
    )
    st.divider()

    # Sidebar Inputs (stored in session_state to keep between reruns)
    if selected_option == "📁 File Upload":
        uploaded_files = st.sidebar.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            key="sidebar_file_upload"
        )
        st.session_state.last_upload = {'type': 'file', 'files': uploaded_files}

    elif selected_option == "🔗 URL/Links":
        urls_text = st.sidebar.text_area(
            "Paste PDF URLs (one per line)",
            height=120,
            key="sidebar_urls"
        )
        st.session_state.last_upload = {'type': 'url', 'urls_text': urls_text}

    elif selected_option == "📦 Zip Upload":
        zip_file = st.sidebar.file_uploader(
            "Choose a zip file of PDFs",
            type=['zip'],
            key="sidebar_zip"
        )
        st.session_state.last_upload = {'type': 'zip', 'zip_file': zip_file}

# Main Header
st.markdown("<h1 style='text-align:center; margin-bottom:0;'>📄 Resume Processing System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:grey; margin-top:0;'>Fast, Reliable, and Modern Resume Parser</h4>", unsafe_allow_html=True)

# Process and Upload Buttons in Main Area
with st.container():
    st.divider()

    # Possible upload options
    if selected_option == "📁 File Upload":
        if st.session_state.last_upload.get('files'):
            if st.button("Process Uploaded Files", key="process_uploaded_main"):
                with st.spinner("Processing uploaded files..."):
                    output_dir = clear_temp_resumes()
                    process_resumes(processor, max_workers)

    elif selected_option == "🔗 URL/Links":
        if st.session_state.last_upload.get('urls_text', "").strip():
            if st.button("Download and Process URLs", key="process_urls_main"):
                with st.spinner("Downloading and processing URLs..."):
                    output_dir = clear_temp_resumes()
                    process_resumes(processor, max_workers)

    elif selected_option == "📦 Zip Upload":
        if st.session_state.last_upload.get('zip_file'):
            if st.button("Extract and Process Zip", key="process_zip_main"):
                with st.spinner("Extracting and processing zip file..."):
                    output_dir = clear_temp_resumes()
                    process_resumes(processor, max_workers)

    # Clear Results button, always visible if data exists
    if st.session_state.processed_data or st.session_state.processing_complete:
        if st.button("🗑️ Clear Results", key="clear_results_main"):
            clear_results(processor)
            st.success("Results cleared. Ready for new processing!")

# Results Area
st.markdown("---")
st.header("📊 Processing Results")

if st.session_state.processing_complete and st.session_state.processed_data:
    results = st.session_state.processed_data

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", len(results))
    with col2:
        st.metric("Success Rate", f"{len(results)}/{len(results)}")
    with col3:
        processing_time = st.session_state.processing_time
        st.metric("Processing Time (s)", f"{processing_time:.2f}" if processing_time is not None else "-")

    # Display DataFrame preview
    df = pd.DataFrame(results)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Download options
    col1, col2, col3, col4 = st.columns(4)
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).apply(lambda x: ILLEGAL_CHARACTERS_RE.sub('', x))

    with col2:
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        st.download_button(
            label="📊 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=f"parsed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col3:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📄 Download as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"parsed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("No results available yet. Use the control panel to select and upload resumes, then process them here.")

# Resume Matching
st.markdown("---")
if st.session_state.processing_complete and st.session_state.processed_data:

    # User inputs/pastes JD
    st.subheader("Match Resumes to Job Description (Relative Candidate Ranking)")
    jd_text = st.text_area("Paste Job Description here", height=120)

    if jd_text.strip() and st.button("Sorting & Ranking Resumes by JD"):
        status_text = st.empty()
        progress = st.progress(0)

        st.session_state["last_ranking_results"] = []
        st.session_state["filtered_candidates"] = None
        dataframe = pd.DataFrame(st.session_state.processed_data)

        # Parsed data to Llamaindex Document & Embeddings
        metadata_fields = ["Name", "Email", "Phone", "Education", "Job Title", "Experience", "resume_path"]
        docs = [
            Document(
                text=row_to_text(row),
                metadata={field: row[field] for field in metadata_fields}
            )
            for _, row in dataframe.iterrows()
        ]

        # Llama-index Config
        docs_dict = {doc.id_: doc for doc in docs}
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=API_KEY)

        status_text.info("Indexing resumes...")
        index = VectorStoreIndex.from_documents(docs)
        progress.progress(0.15)
        
        # Multiquery Generation
        status_text.info("Generating semantic multiqueries from JD...")
        multiqueries = generate_multiqueries(client, job_schema, jd_text, n=4)
        progress.progress(0.30)

        # Scoring resumes
        status_text.info("Computing resume scores...")
        bm25_scores = compute_bm25_filtered_scores(docs, multiqueries)
        jaccard_scores = compute_jaccard_filtered_scores(multiqueries, [doc.text_resource.text for doc in docs])
        node_scores = compute_node_scores(docs, multiqueries, Settings.embed_model)

        raw_scores = (50 * node_scores + 0.3 * bm25_scores + 20 * jaccard_scores)
        progress.progress(0.80)

        valid_idx = [i for i, s in enumerate(raw_scores) if s > MIN_RAW_SCORE]
        if not valid_idx:
            st.warning("No strong resumes found, showing relative ranking of top candidates.")
            valid_idx = list(range(len(docs)))  # fallback mechanism to show all
        else:
            st.info(f"Found {len(valid_idx)} strong resumes, showing relative ranking of top candidates.")

        # Filtering according to high-matching resumes
        docs = [docs[i] for i in valid_idx]
        bm25_scores = np.array(bm25_scores)[valid_idx]
        jaccard_scores = np.array(jaccard_scores)[valid_idx]
        node_scores = np.array(node_scores)[valid_idx]
        raw_scores = np.array(raw_scores)[valid_idx]

        # Normalizing scores for easy visuals
        status_text.info("Normalizing and combining all scores...")
        bm25_norm = normalize(bm25_scores)
        jaccard_norm = normalize(jaccard_scores)
        node_norm = normalize(node_scores)

        final_scores = (0.5 * node_norm + 0.3 * bm25_norm + 0.2 * jaccard_norm) * 100
        final_scores = np.clip(final_scores, 0, 100)
        progress.progress(0.90)

        # Reranking candidates according to combined scores
        status_text.info("Sorting candidates and rendering results...")
        reranked_idx = np.argsort(final_scores)[::-1]
        top_candidates = [docs[i] for i in reranked_idx[:TOP_N]]

        # Storing final results
        results = st.session_state.get("last_ranking_results", [])
        for rank, i in enumerate(reranked_idx[:TOP_N]):
            doc = docs[i]
            meta = doc.metadata
            results.append({
                "Score": final_scores[i],
                "metadata": meta,
                "Resume Text": doc.text_resource.text,
            })

        st.session_state["last_ranking_results"] = results
        st.session_state["last_ranking_jd"] = jd_text
        progress.progress(1.0)

        resume_paths = [doc.metadata['resume_path'] for doc in top_candidates]
        filtered_dataframe = st.session_state.get("filtered_candidates", None)

        filtered_dataframe = dataframe[dataframe['resume_path'].isin(resume_paths)]
        st.session_state["filtered_candidates"] = filtered_dataframe

    # UI for Top Candidates
    if st.session_state.get("last_ranking_results"):
        results = st.session_state.get("last_ranking_results", [])
        filtered_dataframe = st.session_state.get("filtered_candidates", None)

        st.subheader("Relative Candidate Match")
        for candidate in results:
            render_candidate(
                meta=candidate['metadata'],
                score=candidate['Score']
            )

        # Download options
        col1, col2, col3, col4 = st.columns(4)
        ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

        for col in filtered_dataframe.select_dtypes(include=['object']):
            filtered_dataframe[col] = filtered_dataframe[col].astype(str).apply(lambda x: ILLEGAL_CHARACTERS_RE.sub('', x))

        with col2:
            excel_buffer = io.BytesIO()
            filtered_dataframe.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)

            st.download_button(
                label="📊 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"filtered_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            csv_buffer = io.StringIO()
            filtered_dataframe.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📄 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"filtered_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )