import os
from operator import itemgetter
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import create_kv_docstore, LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)

# Load environment variables
load_dotenv("secret.env")
openai_api_key = os.environ["OPENAI_API_KEY"]


def format_docs(data: dict[str, Any]) -> str:
    docs = data["context"]
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(client_input: dict[str, str]) -> str:
    chat_history = client_input["chat_history"]
    return "\n".join([f"user:{user}" + "\n" + f"ai:{ai}" for user, ai in zip(chat_history["user"], chat_history["ai"])])


class QueryDoc:
    """
    Provides functions for document QnA
    """

    def __init__(self, user_file_path: str, embeddings: Embeddings | None = None):
        """
        Instantiate QnA chain
        """
        # Load embedding model
        if not embeddings:
            embeddings = HuggingFaceEmbeddings(
                model_name=config["EMBEDDINGS_MODEL_NAME"],
                cache_folder="data/embedding_models"
            )

        # Instantiate document retriever
        vectorstore = FAISS.load_local(
            f"{user_file_path}/child_vectorstore",
            embeddings=embeddings
        )
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config["PARENT_CHUNK_SIZE"])
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=config["CHILD_CHUNK_SIZE"])
        store = create_kv_docstore(LocalFileStore(f"{user_file_path}/parent_vectorstore"))
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        configurable_retriever = retriever.configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use",
            )
        )

        # Prepare the LLM
        llm = ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=config["LLM_BASE_URL"],
            model=config["MODEL_NAME"],
            temperature=config["MODEL_TEMPERATURE"]
        )

        # Create base rag chain
        prompt_template = """
I want you to act as a document that I am having a conversation with. Your name is "AI Assistant". Using the provided context, answer the user's question to the best of your ability using the resources provided.
If there is nothing in the context relevant to the question at hand, just say "This is beyond the scope of my knowledge!" and stop after that. Refuse to answer any question not about the info. Never break character.
------------
<context>
{context}
</context>
------------

User Query: {question}
        """
        response_prompt = PromptTemplate.from_template(prompt_template)
        base_rag_chain = RunnableParallel(
            {"context": configurable_retriever, "question": RunnablePassthrough()}
        ).assign(answer=(
                RunnablePassthrough.assign(context=format_docs)
                | response_prompt
                | llm
                | StrOutputParser()
        )
        )

        # Create question rephrase chain
        query_rephrase_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. 
DO NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Latest User Question: {question} 
Standalone Question:
        """
        query_rephrase_prompt_template = PromptTemplate.from_template(query_rephrase_prompt)
        query_rephrase_chain = (
                {"chat_history": format_chat_history, "question": itemgetter("question")}
                | query_rephrase_prompt_template | llm | StrOutputParser()
        )

        # Create RAG chain with memory
        self.rag_chain = RunnableBranch(
            (lambda x: len(x["chat_history"]["ai"]) > 0, query_rephrase_chain | base_rag_chain),
            (lambda x: x["question"]) | base_rag_chain
        )

    def run(
            self,
            query: str,
            chat_history: dict[str, list[str]],
            source_docs_to_use: list[str]
    ) -> dict[str, Any]:
        """
        Questions and answers over knowledge base
        """
        # Dynamic search arguments
        search_kwargs = {
            "k": config["TOP_K_CHUNKS"],
            "filter": {"source": source_docs_to_use}
        }

        # Run doc qna chain
        response = self.rag_chain.invoke(
            {"question": query, "chat_history": chat_history},
            config={
                "configurable": {"search_kwargs": search_kwargs},
                'callbacks': [ConsoleCallbackHandler()]
            }
        )

        return response
