from typing import Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from scraper import scrape_webpage
from vector_store import get_vector_store_manager, VectorStoreManager


class IngestionService:
    """
    Ingestion Pipeline:
    fetch webpage -> clean html -> chunk -> embed -> chromadb
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.vector_store_manager = vector_store_manager or get_vector_store_manager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest_url(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the full ingestion pipeline:
        1. Checks if already indexed (skips re-scraping unless force_refresh=True)
        2. Fetches webpage and cleans HTML
        3. Chunks text with overlap
        4. Embeds chunks with nomic-embed-text
        5. Persists vectors into ChromaDB
        """
        url_str = str(url)

        # Check existing index
        if self.vector_store_manager.is_url_indexed(url_str) and not force_refresh:
            existing_count = self.vector_store_manager.get_url_chunks_count(url_str)
            return {
                "status": "already_indexed",
                "url": url_str,
                "title": "Already Indexed",
                "chunks_indexed": existing_count,
                "message": "Webpage is already indexed in ChromaDB. Set force_refresh=True to re-index.",
            }

        # Step 1 & 2: Fetch webpage & clean HTML
        scraped_data = await scrape_webpage(url_str)
        title = scraped_data["title"]
        clean_content = scraped_data["content"]
        char_count = scraped_data.get("char_count", len(clean_content))

        # If re-indexing, remove old records first
        if force_refresh and self.vector_store_manager.is_url_indexed(url_str):
            self.vector_store_manager.delete_url(url_str)

        # Step 3: Chunk text
        chunks = self.text_splitter.split_text(clean_content)
        if not chunks:
            raise ValueError("No text content could be extracted into chunks.")

        # Step 4 & 5: Embed & Store in ChromaDB
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "url": url_str,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        self.vector_store_manager.vector_store.add_documents(documents)

        return {
            "status": "success",
            "url": url_str,
            "title": title,
            "chunks_indexed": len(documents),
            "char_count": char_count,
            "sample_chunk": chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0],
            "message": f"Successfully scraped, chunked, and stored {len(documents)} chunks in ChromaDB.",
        }

