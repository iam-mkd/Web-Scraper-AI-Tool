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
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes the QA pipeline:
        1. Embeds question and queries ChromaDB strictly filtered by URL
        2. Formats top-k retrieved chunks into structured context
        3. Invokes Granite LLM (traced by LangSmith)
        4. Returns answer and source text chunks
        """
        url_str = str(url)

        # Retrieve top-k context chunks filtered strictly by URL
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={
                "filter": {"url": url_str},
                "k": top_k,
            }
        )

        retrieved_docs = await retriever.ainvoke(query)

        if not retrieved_docs:
            return {
                "url": url_str,
                "query": query,
                "answer": f"No indexed content found for URL: {url_str}. Please scrape the webpage first.",
                "sources": [],
                "model": self.llm_model,
            }

        # Build context from chunks
        context_parts = [
            f"[Chunk {i}]\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs, start=1)
        ]
        context_text = "\n\n".join(context_parts)
        page_title = retrieved_docs[0].metadata.get("title", "Webpage")

        # Run inference (automatically traced by LangSmith)
        answer = await self.chain.ainvoke({
            "title": page_title,
            "url": url_str,
            "context": context_text,
            "question": query,
        })

        return {
            "url": url_str,
            "query": query,
            "answer": answer.strip(),
            "sources": [doc.page_content for doc in retrieved_docs],
            "model": self.llm_model,
        }

