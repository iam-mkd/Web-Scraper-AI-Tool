import os
import asyncio
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Configure Hugging Face credentials and warnings
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


class GemmaEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation using SentenceTransformer and google/embeddinggemma-2.
    Embeds documents with prompt_name='Document' and queries with prompt_name='SearchQuery'.
    """
    _model_cache: Dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        device: Optional[str] = None,
        query_prompt_name: str = "SearchQuery",
        doc_prompt_name: str = "Document",
    ):
        self.model_name = model_name
        self.device = device
        self.query_prompt_name = query_prompt_name
        self.doc_prompt_name = doc_prompt_name

        if self.model_name not in self._model_cache:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            self._model_cache[self.model_name] = SentenceTransformer(
                self.model_name,
                device=self.device,
                token=token,
            )
        self.model = self._model_cache[self.model_name]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            prompt_name=self.doc_prompt_name,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            prompt_name=self.query_prompt_name,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store instances and Gemma embeddings.
    """
    _instance: Optional["VectorStoreManager"] = None

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "web_scraper_docs",
        embedding_model: str = "google/embeddinggemma-2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize Gemma SentenceTransformer embeddings
        self.embeddings = GemmaEmbeddings(model_name=self.embedding_model)

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
