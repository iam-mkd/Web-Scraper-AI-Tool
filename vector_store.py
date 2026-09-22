import os
from typing import Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded (e.g. for LangSmith or Ollama configs)
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store instances and embeddings.
    """
    _instance: Optional["VectorStoreManager"] = None

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "web_scraper_docs",
        embedding_model: str = "nomic-embed-text",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Initialize ChromaDB persistent vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def is_url_indexed(self, url: str) -> bool:
        """Check if any chunks for this URL exist in ChromaDB."""
        try:
            results = self.vector_store.get(where={"url": str(url)}, limit=1)
            return bool(results and results.get("ids"))
        except Exception:
            return False

    def get_url_chunks_count(self, url: str) -> int:
        """Count the number of indexed chunks for a given URL."""
        try:
            results = self.vector_store.get(where={"url": str(url)})
            return len(results["ids"]) if (results and results.get("ids")) else 0
        except Exception:
            return 0

    def delete_url(self, url: str) -> None:
        """Delete all chunks belonging to a specific URL."""
        try:
            results = self.vector_store.get(where={"url": str(url)})
            if results and results.get("ids"):
                self.vector_store.delete(ids=results["ids"])
        except Exception as e:
            print(f"Warning: Error while deleting chunks for {url}: {e}")


# Default shared vector store manager instance
default_vector_store_manager = VectorStoreManager()


def get_vector_store_manager(
    persist_directory: str = "./chroma_db",
    collection_name: str = "web_scraper_docs",
) -> VectorStoreManager:
    """
    Returns a VectorStoreManager instance. If default params match, returns the shared instance.
    """
    if persist_directory == "./chroma_db" and collection_name == "web_scraper_docs":
        return default_vector_store_manager
    return VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

