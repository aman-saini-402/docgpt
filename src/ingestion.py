import yaml
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import create_kv_docstore, LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load project settings
with open("setup.yaml", "r") as file:
    config = yaml.load(file, yaml.Loader)


class CreateIndex:
    """Creates vector database of the given set of documents"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _does_vectorstore_exist(self, embeddings: HuggingFaceEmbeddings) -> bool:
        """
        Checks if vectorstore exists
        """
        try:
            FAISS.load_local(
                folder_path=f"{self.file_path}/child_vectorstore",
                embeddings=embeddings
            )
        except RuntimeError:
            return False
        return True

    def process_documents(self, docs_to_ingest: list[str]) -> None:
        """
        Create and store vector database
        """
        # Load open source embeddings
        os_embeddings = HuggingFaceEmbeddings(
            model_name=config["EMBEDDINGS_MODEL_NAME"],
            cache_folder="data/embedding_models"
        )

        # Load files
        print(f"Loading documents from {self.file_path}")
        data = []
        for pdf_file in docs_to_ingest:
            loader = PyMuPDFLoader(f"{self.file_path}/{pdf_file}")
            data += loader.load()

        # Initialize components of parent-doc-retriever
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config["PARENT_CHUNK_SIZE"])
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=config["CHILD_CHUNK_SIZE"])
        store = create_kv_docstore(LocalFileStore(f"{self.file_path}/parent_vectorstore"))

        # Create vector store
        if self._does_vectorstore_exist(os_embeddings):
            # Update and store vectorstore locally
            print(f"Appending to existing vectorstore at {self.file_path}/child_vectorstore")
            vectorstore = FAISS.load_local(
                folder_path=f"{self.file_path}/child_vectorstore",
                embeddings=os_embeddings
            )
        else:
            # Create a new vectorstore
            print("Creating new vectorstore")
            vectorstore = FAISS.from_texts([" "], os_embeddings)

        # Create child embeddings
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        print("Creating embeddings. May take some minutes...")
        retriever.add_documents(data)

        # Store vector database locally
        vectorstore.save_local(f"{self.file_path}/child_vectorstore")
        print("Ingestion complete! You can now query your private data.")
