import glob
import os

import streamlit as st
import yaml
from streamlit_authenticator import Authenticate
from yaml import SafeLoader

from src.rag_pipeline import QueryDoc
from src.utils import deactivate_pdf_view, upload_file_form, show_vector_store, clear_chat, extract_source_docs, \
    prepare_chat_history, post_inference_buttons, log_text_convo, load_embedding_model

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)


def activate_register_user_view() -> None:
    st.session_state["register_user_view"] = True


def deactivate_register_user_view() -> None:
    st.session_state["register_user_view"] = False


def register_new_user() -> None:
    # Register new user
    _, c, _ = st.columns([0.2, 0.6, 0.2], gap="small")
    with c:
        st.warning("Do not use HQ-credentials!", icon="⚠️")
        try:
            _, username_of_registered_user, _ = authenticator.register_user(
                location="main",
                preauthorization=False,
                fields={
                    'Form name': 'Sign Up',
                    'Email': 'Email',
                    'Username': 'Username',
                    'Password': 'Password',
                    'Repeat password': 'Repeat password',
                    'Register': 'Register'}
            )
            if username_of_registered_user:
                with open('auth_creds.yaml', 'w') as x:
                    yaml.dump(auth_file, x, default_flow_style=False)
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

        # Go back to login page
        st.button("Go to login", on_click=deactivate_register_user_view, type="primary")
        st.caption("Click this button to go back")


def do_security_check() -> bool:
    _, c, _ = st.columns([0.2, 0.6, 0.2], gap="small")
    with c:
        _, authentication_status, username = authenticator.login("main")
        if authentication_status:
            authenticator.logout(location="sidebar")
            st.session_state["username"] = username
            return True
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            return False
        elif not authentication_status:
            st.error("username/password is incorrect")
            return False
        else:
            return False


def _save_edits() -> None:
    """
    A helper function to keep persist selected files in uploaded view
    """
    try:
        st.session_state["files_included"] = st.session_state["edited_files_included"]
    except KeyError:
        pass


def pdf_view() -> None:
    """
    Display retrieved text chunks in PDF
    """
    # View description
    st.caption("View text used to answer your query")
    st.divider()

    # Display PDF markdown
    st.markdown(st.session_state["pdf_markdown"], unsafe_allow_html=True)

    # Display metadata(extra info) in sidebar
    doc = st.session_state["last_used_sources"].loc[st.session_state["doc_to_display"]]
    with st.sidebar:
        st.button("Go back", type="primary", on_click=deactivate_pdf_view)
        st.divider()
        st.write("**Source file**: " + doc["source"])
        st.write("**Page No**: " + str(doc["page_num"] + 1))
        st.write("**Chunk Text**")
        with st.container(height=250, border=True):
            st.write(doc["chunk_text"])


def inference_view():
    """
    Space for interacting with LLM and query over knowledge space
    """
    # Show uploaded files in sidebar
    with st.sidebar:
        st.write(f"You're logged in as **{st.session_state['username']}**")
        upload_file_form()
        show_vector_store()

    # Show clear chat option
    c1, c2 = st.columns([0.8, 0.2], gap="small")
    c1.caption("Expand sidebar to manage your documents")
    with c2:
        st.button(
            label="Clear chat",
            on_click=clear_chat,
            disabled=len(st.session_state["messages"]) == 0
        )
    st.divider()

    # Show chat history
    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    # Disable chat if no file in KB
    if len(glob.glob(f'data/{st.session_state["username"]}/*.pdf')) == 0:
        st.info("Upload files to begin querying...")
        st.stop()

    # Load RAG pipeline for current user
    with st.spinner("Loading pipeline..."):
        if "rag_pipeline" not in st.session_state:
            st.session_state["rag_pipeline"] = QueryDoc(
                user_file_path=f'data/{st.session_state["username"]}',
                embeddings=load_embedding_model()
            )
    rag_pipeline = st.session_state["rag_pipeline"]

    # React to user query
    if query := st.chat_input("Enter your Query", max_chars=400):
        st.chat_message("human").write(query)
        # Call RAG
        with st.spinner("Generating response..."):
            response = rag_pipeline.run(
                query=query,
                chat_history=prepare_chat_history(),
                source_docs_to_use=[f'data/{st.session_state["username"]}/{file}' for file in
                                    st.session_state["files_included"]]
            )

        # Show response
        st.chat_message("ai").write(response["answer"])

        # Update chat history
        st.session_state["messages"].append((query, response["answer"]))
        st.session_state["rephrased_input"].append(response["question"])
        log_text_convo()

        # Extract source documents from response
        st.session_state["last_used_sources"] = extract_source_docs(response)

    # Post-inference buttons
    if st.session_state["last_used_sources"] is not None:
        post_inference_buttons()

    # Add disclaimer
    st._bottom.caption(
        "⚠️ Disclaimer: This response was generated using DocGPT, an experimental AI tool \
        that processes and summarizes content from pre-uploaded documents. \
        While we aim for accuracy, the tool may produce errors or incomplete information. \
        Please use these responses with caution and verify critical details independently.")


if __name__ == "__main__":

    # Initialize a session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_used_sources" not in st.session_state:
        st.session_state.last_used_sources = None  # its a dataframe
    if "pdf_view" not in st.session_state:
        st.session_state["pdf_view"] = False
    if "chunk_text_to_display" not in st.session_state:
        st.session_state["chunk_text_to_display"] = ""
    if "rephrased_input" not in st.session_state:
        st.session_state["rephrased_input"] = []
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = []
    if "log_success" not in st.session_state:
        st.session_state["log_success"] = []
    if "register_user_view" not in st.session_state:
        st.session_state["register_user_view"] = False

    # Page Config
    st.set_page_config(page_title="DocGPT v0.2", layout="wide", initial_sidebar_state="auto")
    st.write("### DocGPT v0.2")  # title
    st.markdown("""
        <style>
                .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True
                )

    # Load a common authentication object for both login and sign up
    with open("auth_creds.yaml") as f:
        auth_file = yaml.load(f, Loader=SafeLoader)
    authenticator = Authenticate(
        auth_file["credentials"],
        auth_file["cookie"]["name"],
        auth_file["cookie"]["key"],
        auth_file["cookie"]["expiry_days"],
        auth_file["preauthorized"]
    )

    # Authenticate user
    _, c, _ = st.columns([0.1, 0.8, 0.1])
    with c:
        # Register new user
        if st.session_state["register_user_view"]:
            register_new_user()
            st.stop()

        # Authenticate user
        if not do_security_check():
            # Option to register new user
            _, c2, _ = st.columns([0.2, 0.5, 0.3])
            with c2:
                st.button("New User?", on_click=activate_register_user_view, type="primary")
            st.stop()

    # Initialize list of files in Kb for current user
    if "files_included" not in st.session_state:
        st.session_state["files_included"] = list(
            map(os.path.basename, glob.glob(f'data/{st.session_state["username"]}/*.pdf')))

    if st.session_state["pdf_view"]:
        pdf_view()
    else:
        inference_view()
