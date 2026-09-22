from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from vector_store import get_vector_store_manager, VectorStoreManager


class QAService:
    """
    Question Answering Pipeline:
    embed question -> chroma db -> top k chunks -> granite (LLM) -> answer
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm_model: str = "granite3-dense:2b",
        temperature: float = 0.1,
    ):
        self.vector_store_manager = vector_store_manager or get_vector_store_manager()
        self.llm_model = llm_model

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=temperature,
        )

        # RAG Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a helpful and accurate assistant analyzing a webpage.\n"
                    "Answer the user's question accurately using ONLY the provided webpage context.\n"
                    "When asked to list or identify products or items:\n"
                    "- Extract each distinct product title and its key specifications/price.\n"
                    "- Present each as a single bullet point.\n"
                    "- List each product only once without repeating.\n"
                    "- Once all items in the context are listed, stop and provide a brief summary.\n"
                    "If the context does not contain enough information to answer the question, state that clearly.\n\n"
                    "Webpage Title: {title}\n"
                    "Webpage URL: {url}\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def answer_query(
        self,
        url: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the QA pipeline:
        1. Dynamically determines effective top_k (covers all chunks if page <= 12 chunks)
        2. Embeds question and queries ChromaDB strictly filtered by URL with relevance scores
        3. Sorts context chunks in natural document order
        4. Invokes Granite LLM (traced by LangSmith)
        5. Returns answer and rich source objects with content, metadata (url, chunk_id), and score
        """
        url_str = str(url)
        total_url_chunks = self.vector_store_manager.get_url_chunks_count(url_str)

        # Dynamic retrieval adaptation:
        # If top_k is not explicitly provided:
        # - For single webpages with <= 12 chunks (~2.5k-3k tokens), retrieve all chunks so nothing is missed.
        # - For larger webpages, retrieve top 8 chunks.
        if top_k is None or top_k <= 0:
            if 0 < total_url_chunks <= 12:
                effective_k = total_url_chunks
            else:
                effective_k = min(8, total_url_chunks) if total_url_chunks > 0 else 8
        else:
            effective_k = min(top_k, total_url_chunks) if total_url_chunks > 0 else top_k

        # Retrieve context chunks with relevance scores filtered strictly by URL
        try:
            results_with_scores = await self.vector_store_manager.vector_store.asimilarity_search_with_relevance_scores(
                query=query,
                k=effective_k,
                filter={"url": url_str},
            )
        except Exception:
            # Fallback if relevance score metric calculation is not supported
            raw_results = await self.vector_store_manager.vector_store.asimilarity_search_with_score(
                query=query,
                k=effective_k,
                filter={"url": url_str},
            )
            results_with_scores = [
                (doc, round(1.0 / (1.0 + float(dist)), 4))
                for doc, dist in raw_results
            ]

        if not results_with_scores:
            return {
                "answer": f"No indexed content found for URL: {url_str}. Please scrape the webpage first.",
                "sources": [],
                "url": url_str,
                "query": query,
                "model": self.llm_model,
            }

        # Sort chunks in natural document reading order by chunk_id
        results_with_scores.sort(
            key=lambda item: int(item[0].metadata.get("chunk_id", item[0].metadata.get("chunk_index", 0)))
        )

        # Build context from chunks and prepare detailed sources
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(results_with_scores, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("chunk_index", i - 1))
            score_val = round(float(score), 4)

            context_parts.append(f"[Chunk {chunk_id}]\n{doc.page_content}")

            # Structured metadata ensuring url and chunk_id are cleanly available
            meta = {
                "url": doc.metadata.get("url", url_str),
                "chunk_id": int(chunk_id),
            }
            if "title" in doc.metadata:
                meta["title"] = doc.metadata["title"]
            if "total_chunks" in doc.metadata:
                meta["total_chunks"] = doc.metadata["total_chunks"]

            sources.append({
                "content": doc.page_content,
                "metadata": meta,
                "score": score_val,
            })

        context_text = "\n\n".join(context_parts)
        page_title = results_with_scores[0][0].metadata.get("title", "Webpage")

        # Run inference (automatically traced by LangSmith)
        answer = await self.chain.ainvoke({
            "title": page_title,
            "url": url_str,
            "context": context_text,
            "question": query,
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "url": url_str,
            "query": query,
            "model": self.llm_model,
        }
