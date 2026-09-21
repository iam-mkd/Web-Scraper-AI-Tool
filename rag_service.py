import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Ensure environment variables are loaded for LangSmith tracing
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


class RAGService:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "web_scraper_docs",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "granite3-dense:2b",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Initialize ChromaDB persistent vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0.1,
        )

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def is_url_indexed(self, url: str) -> bool:
        """Check if any chunks for this URL already exist in ChromaDB."""
        try:
            results = self.vector_store.get(where={"url": url}, limit=1)
            return bool(results and results.get("ids"))
        except Exception:
            return False

    def delete_url(self, url: str) -> None:
        """Remove previously indexed chunks for a specific URL."""
        try:
            results = self.vector_store.get(where={"url": url})
            if results and results.get("ids"):
                self.vector_store.delete(ids=results["ids"])
        except Exception as e:
            print(f"Warning: Error while removing previous chunks for {url}: {e}")

    def index_webpage(self, url: str, title: str, content: str, force_refresh: bool = False) -> int:
        """
        Chunks the content, attaches metadata, and indexes into ChromaDB.
        If force_refresh is True, existing entries for this URL are removed first.
        """
        if self.is_url_indexed(url):
            if not force_refresh:
                # Already indexed, fetch count of existing chunks
                existing = self.vector_store.get(where={"url": url})
                return len(existing["ids"]) if (existing and existing.get("ids")) else 0
            else:
                self.delete_url(url)

        # Chunk text
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return 0

        # Build documents
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        # Add to ChromaDB vector store
        self.vector_store.add_documents(documents)
        return len(documents)

    async def query_webpage(self, url: str, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Retrieves relevant chunks for the given URL and generates an answer using granite3-dense:2b.
        Execution is automatically traced via LangSmith if configured in .env.
        """
        # Retrieve context chunks strictly filtered to the requested URL
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "filter": {"url": url},
                "k": k,
            }
        )

        retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            return {
                "answer": f"No indexed content found for URL: {url}. Please scrape the webpage first.",
                "sources": [],
                "model": self.llm_model,
            }

        # Build context from chunks
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(f"[Chunk {i}]\n{doc.page_content}")
        context_text = "\n\n".join(context_parts)

        page_title = retrieved_docs[0].metadata.get("title", "Webpage")

        # Prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a helpful and accurate assistant analyzing a webpage.\n"
                    "Answer the user's question accurately using ONLY the provided webpage context.\n"
                    "If the context does not contain enough information to answer the question, state that clearly.\n\n"
                    "Webpage Title: {title}\n"
                    "Webpage URL: {url}\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        # Run inference (automatically traced by LangSmith)
        answer = await chain.ainvoke({
            "title": page_title,
            "url": url,
            "context": context_text,
            "question": query,
        })

        return {
            "answer": answer.strip(),
            "sources": [doc.page_content for doc in retrieved_docs],
            "model": self.llm_model,
        }

