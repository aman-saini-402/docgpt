import base64
import csv
import glob
import ntpath
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import ocrmypdf
import pandas as pd
import streamlit as st
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from pdfminer.pdfpage import PDFPage

from src.ingestion import CreateIndex
from src.rag_pipeline import QueryDoc

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)


@st.cache_data
def annotate_text_in_pdf(file_path: str, page_number: int, text: str) -> tuple[str, bool]:
    """
    Highlight "text" in the given PDF file
    """
    # Load source file
    doc = fitz.open(f'data/{st.session_state["username"]}/{file_path}')

    # Check PDF size - cannot display files>2mb on streamlit app
    if os.path.getsize(f'data/{st.session_state["username"]}/{file_path}') / 1000000 > 2:
        if page_number == 0:
            doc.select([0, 1, 2])
            page_number = 0
            page = doc[0]
        elif page_number == len(doc) - 1:
            doc.select([page_number - 2, page_number - 1, page_number])
            page_number = 2
            page = doc[2]
        else:
            doc.select([page_number - 1, page_number, page_number + 1])
            page_number = 1
            page = doc[1]
    else:
        page = doc[page_number]

    # Search for text in specified page of the PDF
    text_instances = page.search_for(text, quads=True)

    # Highlight text
    is_annotated = False
    if len(text_instances) > 0:
        for instance in text_instances:
            page.add_highlight_annot(instance)
        is_annotated = True

    # Change pymupdf document object to iobytes
    doc_bytes = doc.tobytes(garbage=4, deflate=True, clean=True)
    base64_pdf = base64.b64encode(doc_bytes).decode("utf-8")

    # Create PDF markdown for streamlit
    pdf_markdown = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_number + 1}" width="1200" height="650" type="application/pdf"></iframe>'

    return pdf_markdown, is_annotated


def activate_pdf_view(doc_to_display: str) -> None:
    """
    Prepare and activate PDF view
    """
    doc = st.session_state.last_used_sources.loc[doc_to_display]
    source_file = doc["source"]
    page_number = int(doc["page_num"])
    text = doc["chunk_text"]

    # Annotate pdf if not done already for the current retrieval
    with st.spinner("Annotating PDF..."):
        st.session_state["pdf_markdown"], is_annotated = annotate_text_in_pdf(source_file, page_number, text)

    if not is_annotated:
        st.warning("Could not highlight text inside PDF!")

    # Activate pdf view
    st.session_state["pdf_view"] = True

    # Update current chunk text
    st.session_state["doc_to_display"] = doc_to_display


def deactivate_pdf_view() -> None:
    st.session_state["pdf_view"] = False


def upload_file_form() -> None:
    """
    Display an upload form and save new files into knowledge base
    """
    # Upload form
    with st.form("KB", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Upload File in Knowledge Base",
            type=["pdf"],
            accept_multiple_files=True
        )
        is_files_uploaded = st.form_submit_button("Upload")

    if is_files_uploaded:
        # Check if user uploaded any file before hitting upload button
        if len(uploaded_files) == 0:
            st.warning("Please select at least one file to upload!", icon="⚠️")
            st.stop()

        # Save uploaded files to user specific folder
        save_folder = f'data/{st.session_state["username"]}'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        docs_to_ingest = []
        for uploaded_file in uploaded_files:
            save_path = Path(save_folder, uploaded_file.name)
            docs_to_ingest.append(save_path)
            with open(save_path, mode="wb") as w:
                w.write(uploaded_file.getvalue())

        # Check for scanned docs
        run_ocr(docs_to_ingest)

        # Ingest new files to vectorstore
        ingestor = CreateIndex(save_folder)
        with st.spinner("Ingesting new document..."):
            ingestor.process_documents([i.name for i in uploaded_files])
            st.session_state["files_included"] = list(
                map(os.path.basename, glob.glob(f'data/{st.session_state["username"]}/*.pdf')))
        st.success("All files are successfully added to Knowledge Base!")
        with st.spinner("Reloading pipeline with latest changes..."):
            # Re-load RAG pipeline with new files
            st.session_state["rag_pipeline"] = QueryDoc(
                user_file_path=f'data/{st.session_state["username"]}',
                embeddings=load_embedding_model()
            )
        st.success("Ingestion complete! You can now query your private data.")


def show_vector_store() -> None:
    """
    Show files in knowledge base
    """
    # Load files list from knowledge base
    files_in_db = list(map(os.path.basename, glob.glob(f'data/{st.session_state["username"]}/*.pdf')))

    # Show editable list of files
    df_files = pd.DataFrame({
        "file_name": files_in_db,
        "included": pd.Series(files_in_db).isin(st.session_state["files_included"])
    })
    df_files_edited = st.data_editor(
        df_files,
        hide_index=True,
        num_rows="fixed",
        disabled=["file_name"],
        column_config={
            "file_name": st.column_config.TextColumn(
                "File Name",
                width="small"
            ),
            "included": st.column_config.CheckboxColumn(
                "Include",
                width="medium",
                help="Filter files for querying"
            )
        }
    )
    st.session_state["files_included"] = pd.Series(files_in_db)[df_files_edited["included"]].to_list()


def post_inference_buttons() -> None:
    """
    Display buttons after one iteration of inference. Two types of buttons are shown:
    1. Feedback buttons
    2. Link to retrieval text - pressing them will take user to PDF view
    """
    # Take feedback and log user convo
    if len(st.session_state["messages"]) != len(st.session_state["feedback"]):
        c1, c2, _ = st.columns([0.08, 0.08, 0.84], gap="small")
        with c1:
            st.button("👍", on_click=collect_feedback, args=[True])
        with c2:
            st.button("👎", on_click=collect_feedback, args=[False])

    # Show retrieved texts in PDF
    if len(st.session_state["last_used_sources"]) > 0:  # at least one chunk retrieved!
        buttons_width = [0.6 / len(st.session_state["last_used_sources"])] * len(
            st.session_state["last_used_sources"])
        row = st.columns(buttons_width + [1 - np.sum(buttons_width)], gap="small")
        for i, col in enumerate(row):
            if i < len(buttons_width):
                with col:
                    st.button(f"Reference {i + 1}", on_click=activate_pdf_view, args=[f"doc_{i + 1}"])


def extract_source_docs(response: dict[str, Any]) -> pd.DataFrame:
    """
    Extract source documents used to answer user query from the LLM's response
    """
    source = []
    page_no = []
    text = []
    for doc in response["context"]:
        source.append(doc.metadata["source"].split("/")[-1])
        page_no.append(doc.metadata["page"])
        text.append(doc.page_content)

    # Return as a pandas df
    return pd.DataFrame({
        "source": source,
        "page_num": page_no,
        "chunk_text": text
    }, index=[f"doc_{i + 1}" for i in range(len(source))])


def clear_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["last_used_sources"] = None
    st.session_state["rephrased_input"] = []
    st.session_state["feedback"] = []
    st.session_state["log_success"] = []


def prepare_chat_history() -> dict[str, list[str]]:
    """
    Format chat history as required by the chat_history parameter of RAG pipeline
    """
    if config["CHAT_HISTORY_WINDOW"] == 0:
        return {"user": [], "ai": []}

    return {"user": [i[0] for i in st.session_state["messages"][-config["CHAT_HISTORY_WINDOW"]:]],
            "ai": [i[1] for i in st.session_state["messages"][-config["CHAT_HISTORY_WINDOW"]:]]}


def log_text_convo() -> None:
    """
    Save a less detailed chat history with only inputs and outputs
    """
    # Check if the log file exists
    if not os.path.exists("conversations.csv"):
        with open("conversations.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["username", "time", "user_input", "rephrased_input", "output"])
            writer.writeheader()

    with open("conversations.csv", 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["username", "time", "user_input", "rephrased_input", "output"])
        writer.writerow({
            'username': st.session_state["username"],
            'time': datetime.now(),
            'user_input': st.session_state["messages"][-1][0],
            "rephrased_input": st.session_state["rephrased_input"][-1],
            'output': st.session_state["messages"][-1][1]
        })


def log_convo() -> None:
    """
    Function to save chat history to a CSV file
    """
    # get log file name
    filename = "chat_log/chat_log" + f"_{date.today()}" + ".csv"
    fieldnames = ["username", "time", "user_input", "rephrased_input", "output", "feedback", "model_name",
                  "model_temperature", "chat_history_window", "embedding_model"]

    # Check if the log file exists
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # log inputs
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if len(st.session_state["log_success"]) == (len(st.session_state["messages"]) - 1):
            # All previous interactions were successfully logged
            writer.writerow({
                'username': st.session_state["username"],
                'time': datetime.now(),
                'user_input': st.session_state["messages"][-1][0],
                "rephrased_input": st.session_state["rephrased_input"][-1],
                'output': st.session_state["messages"][-1][1],
                'feedback': st.session_state["feedback"][-1],
                'model_name': config["MODEL_NAME"],
                'model_temperature': config["MODEL_TEMPERATURE"],
                'chat_history_window': config["CHAT_HISTORY_WINDOW"],
                'embedding_model': config["EMBEDDINGS_MODEL_NAME"]
            })
            st.session_state["log_success"].append(1)
        elif (len(st.session_state["messages"])) == len(st.session_state["feedback"]):
            # User didn't give feedback in previous messages but finally gives one
            logged = len(st.session_state["log_success"])
            to_log = len(st.session_state["feedback"]) - logged
            for i in range(logged, logged + to_log):
                writer.writerow({
                    'username': st.session_state["username"],
                    'time': datetime.now(),
                    'user_input': st.session_state["messages"][i][0],
                    "rephrased_input": st.session_state["rephrased_input"][i],
                    'output': st.session_state["messages"][i][1],
                    'feedback': st.session_state["feedback"][i],
                    'model_name': config["MODEL_NAME"],
                    'model_temperature': config["MODEL_TEMPERATURE"],
                    'chat_history_window': config["CHAT_HISTORY_WINDOW"],
                    'embedding_model': config["EMBEDDINGS_MODEL_NAME"]
                })
                st.session_state["log_success"].append(1)
        else:
            # User didn't give feedback in previous messages and one of the next message didn't require feedback
            logged = len(st.session_state["log_success"])
            to_log = len(st.session_state["messages"]) - logged
            for i in range(logged, logged + int(to_log)):
                writer.writerow({
                    'username': st.session_state["username"],
                    'time': datetime.now(),
                    'user_input': st.session_state["messages"][i][0],
                    'rephrased_input': st.session_state["rephrased_input"][i],
                    'output': st.session_state["messages"][i][1],
                    'feedback': "NaN",
                    'model_name': config["MODEL_NAME"],
                    'model_temperature': config["MODEL_TEMPERATURE"],
                    'chat_history_window': config["CHAT_HISTORY_WINDOW"],
                    'embedding_model': config["EMBEDDINGS_MODEL_NAME"]
                })
                st.session_state["log_success"].append(1)


def collect_feedback(feedback: bool) -> None:
    # Check if user gave feedback the last time
    x = (len(st.session_state["messages"]) - 1) - len(st.session_state["feedback"])
    if x == 0:
        st.session_state["feedback"].append(feedback)
    else:
        st.session_state["feedback"].extend(["NaN"] * int(x))
        st.session_state["feedback"].append(feedback)
    log_convo()


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> Embeddings:
    """
    Load a common embedding model for all users and cache it with streamlit
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config["EMBEDDINGS_MODEL_NAME"],
        cache_folder="data/embedding_models"
    )
    return embeddings


def run_ocr(docs_to_ingest: list[str]) -> None:
    # Find non-searchable files among uploads
    to_ocr = []
    for doc in docs_to_ingest:
        with open(doc, 'rb') as infile:
            for page in PDFPage.get_pages(infile, check_extractable=False):
                if 'Font' not in page.resources.keys():
                    to_ocr.append(doc)
                    break

    if len(to_ocr) > 0:
        ocr_folder = f'data/{st.session_state["username"]}/scanned_uploads'
        if not os.path.isdir(ocr_folder):
            os.makedirs(ocr_folder)

        for doc in to_ocr:
            # Move scanned uploads to "scanned_uploads" folder
            shutil.move(doc, ocr_folder)
            scanned_doc_path = os.path.join(ocr_folder, ntpath.basename(doc))

            # Perform OCR
            with st.spinner("Found scanned PDFs - performing OCR..."):
                try:
                    ocrmypdf.ocr(
                        scanned_doc_path, doc,
                        output_type="pdf",
                        jobs=6,
                        rotate_pages=True,
                        skip_text=True,
                        optimize=0
                    )
                except ocrmypdf.exceptions.EncryptedPdfError:
                    st.warning(f"OCR failed - found encrypted file ({ntpath.basename(doc)})")
                    # move this file back to normal uploads
                    shutil.move(scanned_doc_path, f'data/{st.session_state["username"]}')
